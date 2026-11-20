"""Concurrency control for GPT Researcher.

The research pipeline fans out at four levels (report sections, sub-queries,
URLs, context chunks). Tasks are cheap - a suspended coroutine is a few KB and
holds no external resource - so wide fan-out is fine. The scarce things are the
*quotas* behind those tasks: search API keys, target websites, LLM token
budgets, sockets and, because block scrapers run in threads, the interpreter's
thread supply. Those are capped here, process-wide, as close to the resource as
possible.

Three building blocks:

``AsyncLimiter``
    Caps how many operations of one kind may be in flight, and optionally how
    fast they may start (token bucket). A limiter is the only thing in this
    package that knows about slots.

``limits``
    A per-event-loop registry of named limiters. Two independent
    ``GPTResearcher`` instances (multi-agent sections, deep-research branches)
    share one budget instead of each getting their own - which is the whole
    point, since a per-instance semaphore multiplies with every section.

``budget`` / ``bounded_gather``
    Per-unit fan-out policy: how many children one coroutine may create, and
    how many URLs one research unit may fetch in total. This is *not* a
    resource limit; it bounds an LLM-chosen plan, which is untrusted input.

Tuning is environment-only, matching ``JEV_CONCURRENCY`` / ``SUMMARY_CONCURRENCY``
/ ``FIRECRAWL_CONCURRENCY``::

    SCRAPER_GLOBAL_CONCURRENCY   process-wide page fetches      (15)
    SEARCH_CONCURRENCY           concurrent searches per provider (8)
    SEARCH_RPM                   searches per minute per provider (0 = off)
    LLM_CONCURRENCY              concurrent LLM calls            (12)
    LLM_TPM                      input tokens per minute (0 = off)
    SUMMARY_CONCURRENCY          concurrent summary calls       (4)
    SUBQUERY_CONCURRENCY         sub-queries per research pass  (0 = off)
    MAX_URLS_PER_TASK            page fetches per research unit (120)
    MAX_SUBQUERIES               sub-queries per research pass  (8)
    MAX_SECTIONS                 report sections per run        (5)

Every knob accepts 0 to disable it.
"""

from . import limits, settings
from .budget import TaskBudget, budget_scope, current_budget
from .bounded import bounded_gather
from .limiter import AsyncLimiter, LimitTimeout, estimate_tokens
from .limits import get, reset, snapshot

__all__ = [
    "AsyncLimiter",
    "LimitTimeout",
    "TaskBudget",
    "bounded_gather",
    "budget_scope",
    "current_budget",
    "estimate_tokens",
    "get",
    "limits",
    "reset",
    "settings",
    "snapshot",
]
