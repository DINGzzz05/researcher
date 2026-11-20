import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from ..concurrency import limits
from .rate_limiter import get_global_rate_limiter


class WorkerPool:
    def __init__(self, max_workers: int, rate_limit_delay: float = 0.0):
        """
        Initialize WorkerPool with concurrency and rate limiting.

        Args:
            max_workers: Maximum number of concurrent workers
            rate_limit_delay: Minimum seconds between requests GLOBALLY (0 = no limit)
                             This delay is enforced across ALL WorkerPools to prevent
                             overwhelming rate-limited APIs.
                             Example: 6.0 for 10 req/min (Firecrawl free tier)

        Note:
            The rate_limit_delay is enforced GLOBALLY using a singleton rate limiter.
            This means if you have multiple GPTResearcher instances (e.g., in deep research),
            they will all share the same rate limit, preventing API overload.
        """
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = asyncio.Semaphore(max_workers)

        # Configure the global rate limiter
        # All WorkerPools share the same rate limiter instance
        global_limiter = get_global_rate_limiter()
        global_limiter.configure(rate_limit_delay)

    @asynccontextmanager
    async def throttle(self):
        """
        Throttle requests with concurrency limits and GLOBAL rate limiting.

        Three gates, outermost first:

        1. this pool's semaphore - how many of *this* researcher's fetches run
           at once (the width that matches its thread pool),
        2. the process-wide ``scraper`` limiter - the real ceiling, shared by
           every pool in the process (multi-agent sections, deep-research
           branches), so a new researcher adds queueing, not capacity,
        3. the global rate limiter - minimum spacing between requests across
           all pools.

        Without (2) the cap multiplies with the number of researchers: three
        sections each with 15 workers meant 45 simultaneous fetches against the
        same search quota and the same egress IP.
        """
        async with self.semaphore:
            async with limits.get("scraper").slot():
                # Use global rate limiter (shared across all WorkerPools)
                global_limiter = get_global_rate_limiter()
                await global_limiter.wait_if_needed()
                yield
