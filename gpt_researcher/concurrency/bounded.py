"""Fan-out with a bounded width and predictable failure semantics.

``asyncio.gather`` has two properties this pipeline cannot live with:

* it starts every child at once (fine for tasks, wrong when each child opens a
  socket), and
* on the first exception it returns early but does **not** cancel its siblings,
  so a failed sub-query leaves the rest scraping and burning quota.

``bounded_gather`` keeps the shape callers already use (a list in child order,
``return_exceptions`` semantics) while fixing both. The width limit here is a
*local* policy - per research pass, per section - not a resource cap. Resource
caps belong in :mod:`gpt_researcher.concurrency.limits`, at the leaf.

The child task is created only after a slot is taken. Creating it eagerly and
awaiting it behind a semaphore would not bound anything: a Task starts running
as soon as it is created, so all children would be in flight while the
semaphore only serialised the awaits.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Iterable


async def bounded_gather(
    awaitables: Iterable[Awaitable[Any]],
    limit: int = 0,
    *,
    return_exceptions: bool = True,
) -> list[Any]:
    """Await ``awaitables`` with at most ``limit`` running at once.

    Args:
        awaitables: Coroutines or futures; each becomes a Task once it may run.
        limit: Maximum concurrent children. 0 or negative means no local cap
            (leaf limiters still apply).
        return_exceptions: When True (default) a failing child yields its
            exception in place, so one bad sub-query cannot discard the others.
            When False the first failure cancels the remaining children - the
            ``TaskGroup`` behaviour - and is re-raised.

    Returns:
        Results in the same order as the inputs.
    """
    items = list(awaitables)
    if not items:
        return []

    limit = int(limit or 0)
    semaphore = asyncio.Semaphore(limit) if limit > 0 else None
    # Filled lazily: None means "not started yet", so cleanup knows which
    # coroutines still need closing to avoid "never awaited" warnings.
    started: list["asyncio.Future[Any] | None"] = [None] * len(items)

    async def run(index: int, awaitable: Awaitable[Any]) -> Any:
        if semaphore is not None:
            async with semaphore:
                task = asyncio.ensure_future(awaitable)
                started[index] = task
                return await task
        task = asyncio.ensure_future(awaitable)
        started[index] = task
        return await task

    runners = [
        asyncio.ensure_future(run(index, awaitable))
        for index, awaitable in enumerate(items)
    ]

    try:
        return await asyncio.gather(
            *runners, return_exceptions=return_exceptions
        )
    except BaseException:
        # Cancelled from outside, or one child failed with
        # return_exceptions=False. Neither gather nor a wrapper task cancels the
        # inner task it was awaiting, so cancel both layers explicitly.
        for runner in runners:
            runner.cancel()
        for task in started:
            if task is not None:
                task.cancel()

        await asyncio.gather(*runners, return_exceptions=True)
        await asyncio.gather(
            *(task for task in started if task is not None), return_exceptions=True
        )
        for index, awaitable in enumerate(items):
            if started[index] is None and hasattr(awaitable, "close"):
                awaitable.close()
        raise
