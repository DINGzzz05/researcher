"""Runtime guards for tool calls: step budget, duplicate calls, oversized results.

The agent decides how many searches to run and with which queries. Nothing in
the model loop bounds those decisions, so one bad plan can burn a run's whole
API budget on the same query. This module adds three limits at the one place
every retriever call passes through:

* **step budget** - each executed tool call costs one step; once the run's
  allowance is spent, further calls are skipped instead of executed,
* **duplicate calls** - an identical ``(provider, query)`` call is served from
  the run's cache instead of hitting the provider again,
* **result truncation** - a single oversized result field is clipped, so one
  tool response cannot dominate the context that follows.

The guard travels in a contextvar, like the concurrency budget, so it follows
the async call tree without being threaded through every signature.
"""

from __future__ import annotations

import copy
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterable

logger = logging.getLogger(__name__)

#: One executed tool call is one step.
DEFAULT_MAX_CALLS = 200
#: Extra executions allowed for an identical call before it is served from cache.
DEFAULT_ALLOW_REPEAT = 0
#: Characters kept from any single string field of a tool result. 0 disables.
DEFAULT_MAX_RESULT_CHARS = 20_000

_TRUNCATED_MARK = " …[truncated {n} chars]"

_guard: ContextVar["RunGuard | None"] = ContextVar("gpt_researcher_run_guard", default=None)


@dataclass
class GuardDecision:
    """What the guard says about one pending tool call."""

    allowed: bool
    duplicate: bool = False
    cached: list | None = None
    reason: str = ""


@dataclass
class RunGuard:
    """Per-run tool-call budget, duplicate cache and result truncator."""

    max_calls: int = DEFAULT_MAX_CALLS
    allow_repeat: int = DEFAULT_ALLOW_REPEAT
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS
    calls: int = 0
    deduped: int = 0
    truncated: int = 0
    _seen: dict[str, int] = field(default_factory=dict)
    _cache: dict[str, Any] = field(default_factory=dict)

    def begin(self, key: str | None) -> GuardDecision:
        """Reserve a step for a tool call, or refuse it when the budget is spent."""
        if self.max_calls > 0 and self.calls >= self.max_calls:
            return GuardDecision(False, reason=f"step budget of {self.max_calls} reached")

        if key is not None and key in self._cache:
            seen = self._seen.get(key, 0)
            if seen > self.allow_repeat:
                self.deduped += 1
                return GuardDecision(True, duplicate=True, cached=copy.deepcopy(self._cache[key]))

        self.calls += 1
        if key is not None:
            self._seen[key] = self._seen.get(key, 0) + 1
        return GuardDecision(True)

    def finish(self, key: str | None, results: Any) -> Any:
        """Cache the first (already clipped) result for ``key`` and return it."""
        if isinstance(results, list):
            results = self.truncate(copy.deepcopy(results))
            if key is not None:
                self._cache.setdefault(key, copy.deepcopy(results))
        return results

    def truncate(self, results: list) -> list:
        if self.max_result_chars <= 0:
            return results
        clipped: list = []
        for item in results:
            shorter = _truncate_item(item, self.max_result_chars)
            if shorter != item:
                self.truncated += 1
            clipped.append(shorter)
        return clipped

    def stats(self) -> dict:
        return {
            "tool_calls": self.calls,
            "duplicate_calls_served": self.deduped,
            "results_truncated": self.truncated,
            "max_calls": self.max_calls,
        }


def _truncate_item(item: Any, limit: int) -> Any:
    """Clip every string field of one tool result to ``limit`` characters."""
    if not isinstance(item, dict):
        return item
    clipped = {}
    for key, value in item.items():
        if isinstance(value, str) and len(value) > limit:
            removed = len(value) - limit
            clipped[key] = value[:limit] + _TRUNCATED_MARK.format(n=removed)
        else:
            clipped[key] = value
    return clipped


def guard_key(provider: str, query: str | None, max_results: int | None = None) -> str | None:
    """Stable identity of a retriever call, or ``None`` when the query is unknown.

    ``max_results`` is part of the identity: the same query asked for ten
    results is not the same call as the same query asked for five.
    """
    if not query:
        return None
    parts = [provider.lower(), " ".join(str(query).lower().split())]
    if max_results is not None:
        parts.append(str(max_results))
    return ":".join(parts)


def current_guard() -> "RunGuard | None":
    """The guard for the current run, or ``None`` when guards are off."""
    return _guard.get()


@contextmanager
def guard_scope(guard: "RunGuard | None" = None):
    """Install ``guard`` for the duration of the research unit."""
    token = _guard.set(guard)
    try:
        yield guard
    finally:
        _guard.reset(token)


def guard_from_config(cfg) -> "RunGuard | None":
    """Build a guard from config, or ``None`` when every limit is disabled."""
    max_calls = int(getattr(cfg, "max_tool_calls", DEFAULT_MAX_CALLS) or 0)
    allow_repeat = int(getattr(cfg, "max_duplicate_tool_calls", DEFAULT_ALLOW_REPEAT) or 0)
    max_chars = int(getattr(cfg, "max_tool_result_chars", DEFAULT_MAX_RESULT_CHARS) or 0)
    if max_calls <= 0 and max_chars <= 0:
        return None
    return RunGuard(max_calls=max_calls, allow_repeat=allow_repeat, max_result_chars=max_chars)
