"""Process-wide registry of named limiters, one set per event loop.

Why a registry and not a module-level ``Semaphore``: an ``asyncio.Semaphore`` is
bound to the loop where it first blocks, and this package is imported by code
that may be driven by ``asyncio.run`` (tests), uvicorn, or a worker loop. Keying
by loop keeps each run isolated without leaking state between them.

Why a registry and not per-instance semaphores: every ``GPTResearcher`` builds
its own ``BrowserManager``/``WorkerPool``, and multi-agent sections plus deep
research branches create several researchers in one run. A per-instance cap is
therefore multiplied by the number of researchers - 3 sections x 15 workers
meant 45 concurrent fetches against the same search quota. Sharing one limiter
per process restores the intended ceiling.

Lookup is by name, and the name chooses the spec from
:mod:`gpt_researcher.concurrency.settings`. ``search:<provider>`` gets its own
limiter per provider, because quotas are per key.
"""

from __future__ import annotations

import asyncio
import weakref

from .limiter import AsyncLimiter
from .settings import LimiterSpec, load

#: WeakKeyDictionary: a closed loop's limiters are collected with it.
_LIMITERS: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, AsyncLimiter]]" = (
    weakref.WeakKeyDictionary()
)


def _spec_for(name: str) -> LimiterSpec:
    settings = load()
    if name == "scraper":
        return settings.scraper
    if name == "search" or name.startswith("search:"):
        # Per-provider isolation, same numbers: SEARCH_CONCURRENCY is a
        # per-key cap, not a global one.
        return settings.search
    if name == "llm":
        return settings.llm
    if name == "llm:summarize":
        return settings.llm_summarize
    return LimiterSpec(name=name)


def get(name: str) -> AsyncLimiter:
    """The limiter called ``name`` for the running loop.

    The first caller fixes the limits; later callers get the same object even
    if the environment changed, so a run cannot acquire more budget part-way
    through. Call :func:`reset` to rebuild (tests, or after a config reload).
    """
    loop = asyncio.get_running_loop()
    table = _LIMITERS.setdefault(loop, {})
    limiter = table.get(name)
    if limiter is None:
        spec = _spec_for(name)
        limiter = AsyncLimiter(
            name,
            max_concurrency=spec.max_concurrency,
            rate_per_second=spec.rate_per_second,
            burst=spec.burst,
        )
        table[name] = limiter
    return limiter


def reset(loop: asyncio.AbstractEventLoop | None = None) -> None:
    """Drop limiters so the next ``get`` rebuilds them from the environment."""
    if loop is None:
        _LIMITERS.clear()
    else:
        _LIMITERS.pop(loop, None)


def snapshot() -> dict[str, dict[str, float | int | str]]:
    """Stats for every limiter used by the running loop."""
    loop = asyncio.get_running_loop()
    return {name: limiter.stats() for name, limiter in _LIMITERS.get(loop, {}).items()}
