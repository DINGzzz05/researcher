"""Environment-driven tunables for the concurrency layer.

Read on demand rather than cached: the values are only fetched when a limiter is
first created (once per event loop) or when a research unit starts, so the cost
is irrelevant, and tests can change a knob with ``monkeypatch.setenv`` without
clearing a cache. A value of 0 (or a malformed value) disables that knob, which
is also the pre-existing behaviour for the older ``*_CONCURRENCY`` variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(float(raw.strip()))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class LimiterSpec:
    """How one named limiter should be built."""

    name: str
    max_concurrency: int = 0
    rate_per_second: float = 0.0
    burst: int = 0

    @property
    def enabled(self) -> bool:
        return self.max_concurrency > 0 or self.rate_per_second > 0


@dataclass(frozen=True)
class ConcurrencySettings:
    scraper: LimiterSpec
    search: LimiterSpec
    llm: LimiterSpec
    llm_summarize: LimiterSpec
    subquery_concurrency: int
    max_urls_per_task: int
    max_subqueries: int
    max_sections: int


def load() -> ConcurrencySettings:
    """Current settings, resolved from the environment."""
    # The scraper cap defaults to the per-pool width so a single-researcher run
    # is unchanged; the difference only shows up when several researchers run
    # at once, which is exactly where the old per-instance cap multiplied.
    scraper_cap = _int(
        "SCRAPER_GLOBAL_CONCURRENCY", _int("MAX_SCRAPER_WORKERS", 15)
    )
    search_concurrency = _int("SEARCH_CONCURRENCY", 8)
    search_rpm = _float("SEARCH_RPM", 0.0)
    llm_concurrency = _int("LLM_CONCURRENCY", 12)
    llm_tpm = _float("LLM_TPM", 0.0)

    return ConcurrencySettings(
        scraper=LimiterSpec("scraper", max(0, scraper_cap)),
        search=LimiterSpec(
            "search",
            max(0, search_concurrency),
            # RPM only matters as a start rate; the semaphore already bounds
            # how many can be open at once.
            rate_per_second=search_rpm / 60.0 if search_rpm > 0 else 0.0,
            burst=max(1, search_concurrency) if search_rpm > 0 else 0,
        ),
        llm=LimiterSpec(
            "llm",
            max(0, llm_concurrency),
            rate_per_second=llm_tpm / 60.0 if llm_tpm > 0 else 0.0,
            # One full minute of tokens may be spent at once, then refill.
            burst=int(llm_tpm) if llm_tpm > 0 else 0,
        ),
        llm_summarize=LimiterSpec(
            "llm:summarize", max(0, _int("SUMMARY_CONCURRENCY", 4))
        ),
        subquery_concurrency=max(0, _int("SUBQUERY_CONCURRENCY", 0)),
        max_urls_per_task=max(0, _int("MAX_URLS_PER_TASK", 120)),
        max_subqueries=max(0, _int("MAX_SUBQUERIES", 8)),
        max_sections=max(0, _int("MAX_SECTIONS", 5)),
    )


def subquery_limit() -> int:
    """Per-pass sub-query fan-out cap (0 = unlimited)."""
    return load().subquery_concurrency
