"""A concurrency limiter with an optional start-rate (token bucket).

``AsyncLimiter`` answers two questions at once:

* how many operations of this kind may be in flight right now (semaphore), and
* how fast new ones may start (token bucket, optional).

Both are needed and neither replaces the other. A semaphore alone lets a burst
start the instant a slot frees; a rate limit alone lets arbitrarily many
connections sit open, which exhausts file descriptors without ever exceeding the
request rate. ``cost`` is expressed in whatever the bucket measures: requests
for search scrapers, prompt tokens for the LLM.

Everything is bound to the event loop that called ``acquire`` first - the
registry in :mod:`gpt_researcher.concurrency.limits` creates one limiter per
loop, so a semaphore is never shared across loops.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager


class LimitTimeout(TimeoutError):
    """A slot could not be acquired within the caller's deadline."""


def estimate_tokens(payload: object) -> float:
    """Cheap prompt-size estimate: ~4 characters per token.

    Only used to weigh LLM calls against ``LLM_TPM``; exact accounting is the
    provider's job. Accepts a string or a chat-message list.
    """
    if payload is None:
        return 1.0
    if isinstance(payload, str):
        text = payload
    elif isinstance(payload, (list, tuple)):
        text = " ".join(str(item) for item in payload)
    else:
        text = str(payload)
    return max(1.0, len(text) / 4.0)


class TokenBucket:
    """Refill-on-read token bucket, safe for concurrent waiters."""

    def __init__(self, rate_per_second: float, burst: int | float):
        self.rate = float(rate_per_second)
        # A request may not wait for more than a full bucket; clamping keeps a
        # single oversized call (e.g. a huge summarization prompt) from
        # deadlocking behind a capacity it can never reach.
        self.capacity = float(max(1.0, burst or self.rate))
        self._tokens = self.capacity
        self._updated: float | None = None
        self._lock = asyncio.Lock()

    async def acquire(self, cost: float = 1.0, timeout: float | None = None) -> None:
        loop = asyncio.get_running_loop()
        cost = min(max(float(cost), 0.0), self.capacity)
        if cost <= 0:
            return
        deadline = None if timeout is None else loop.time() + timeout

        while True:
            async with self._lock:
                now = loop.time()
                if self._updated is None:
                    self._updated = now
                self._tokens = min(
                    self.capacity,
                    self._tokens + (now - self._updated) * self.rate,
                )
                self._updated = now
                if self._tokens >= cost:
                    self._tokens -= cost
                    return
                wait = (cost - self._tokens) / self.rate

            if deadline is not None:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise LimitTimeout(
                        f"token bucket: {cost:g} tokens not available within "
                        f"{timeout:g}s"
                    )
                wait = min(wait, remaining)

            await asyncio.sleep(wait)


class AsyncLimiter:
    """Concurrency + start-rate cap for one named kind of operation."""

    def __init__(
        self,
        name: str,
        max_concurrency: int = 0,
        rate_per_second: float = 0.0,
        burst: int = 0,
    ):
        self.name = name
        self.max_concurrency = max(0, int(max_concurrency or 0))
        self._semaphore = (
            asyncio.Semaphore(self.max_concurrency)
            if self.max_concurrency > 0
            else None
        )
        self._bucket = (
            TokenBucket(rate_per_second, burst or self.max_concurrency or 1)
            if rate_per_second and rate_per_second > 0
            else None
        )
        self._active = 0
        self._peak = 0
        self._acquired = 0
        self._waits = 0
        self._wait_seconds = 0.0
        self._rejected = 0

    # -- introspection -----------------------------------------------------

    @property
    def active(self) -> int:
        return self._active

    def stats(self) -> dict[str, float | int | str]:
        """Snapshot for logging and tests; cheap enough to call per request."""
        return {
            "name": self.name,
            "max_concurrency": self.max_concurrency,
            "active": self._active,
            "peak": self._peak,
            "acquired": self._acquired,
            "waits": self._waits,
            "wait_seconds": round(self._wait_seconds, 4),
            "rejected": self._rejected,
        }

    # -- acquire / release -------------------------------------------------

    async def acquire(self, cost: float = 1.0, timeout: float | None = None) -> None:
        loop = asyncio.get_running_loop()
        started = loop.time()
        holding_semaphore = False

        try:
            if self._semaphore is not None:
                if timeout is None:
                    await self._semaphore.acquire()
                else:
                    try:
                        await asyncio.wait_for(self._semaphore.acquire(), timeout)
                    except (asyncio.TimeoutError, TimeoutError) as exc:
                        self._rejected += 1
                        raise LimitTimeout(
                            f"{self.name}: no slot within {timeout:g}s"
                        ) from exc
                holding_semaphore = True

            if self._bucket is not None:
                await self._bucket.acquire(cost, timeout)
        except BaseException:
            # A rate-limit rejection (or cancellation while waiting for the
            # bucket) must not leak the concurrency slot.
            if holding_semaphore:
                self._semaphore.release()
            raise

        waited = loop.time() - started
        self._active += 1
        self._acquired += 1
        self._peak = max(self._peak, self._active)
        if waited > 1e-4:
            self._waits += 1
            self._wait_seconds += waited

    def release(self) -> None:
        if self._active > 0:
            self._active -= 1
        if self._semaphore is not None:
            self._semaphore.release()

    @asynccontextmanager
    async def slot(self, cost: float = 1.0, timeout: float | None = None):
        """Hold one slot for the duration of the block."""
        await self.acquire(cost=cost, timeout=timeout)
        try:
            yield self
        finally:
            self.release()
