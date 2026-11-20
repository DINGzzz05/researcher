"""Behavioural tests for the concurrency control layer.

No network, no LLM: every test uses either the limiter primitives directly or a
fake provider/retriever. The properties under test are the ones that make the
layer worth having:

* a cap is a cap, not a suggestion (peak in-flight never exceeds it),
* slots are released on exceptions and on cancellation,
* limits are shared across researchers in one process, not per instance,
* a model-chosen plan cannot create unbounded work (budget truncation),
* one failing sub-query does not take its siblings down with it.
"""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from gpt_researcher.concurrency import limits
from gpt_researcher.concurrency.bounded import bounded_gather
from gpt_researcher.concurrency.budget import TaskBudget, budget_scope, current_budget
from gpt_researcher.concurrency.limiter import AsyncLimiter, LimitTimeout


@pytest.fixture(autouse=True)
def clean_registry():
    limits.reset()
    yield
    limits.reset()


async def _tracked_work(limiter, peak, delay=0.01):
    async with limiter.slot():
        peak["active"] += 1
        peak["peak"] = max(peak["peak"], peak["active"])
        try:
            await asyncio.sleep(delay)
        finally:
            peak["active"] -= 1


def test_limiter_caps_inflight_operations():
    async def main():
        limiter = AsyncLimiter("test", max_concurrency=3)
        peak = {"active": 0, "peak": 0}
        await asyncio.gather(*(_tracked_work(limiter, peak) for _ in range(12)))
        assert peak["peak"] <= 3
        assert limiter.stats()["acquired"] == 12
        assert limiter.active == 0

    asyncio.run(main())


def test_limiter_releases_slot_on_exception():
    async def main():
        limiter = AsyncLimiter("test", max_concurrency=1)

        async def boom():
            async with limiter.slot():
                raise ValueError("boom")

        with pytest.raises(ValueError):
            await boom()

        assert limiter.active == 0
        # A leaked slot would make this block forever.
        async with asyncio.timeout(1):
            async with limiter.slot():
                pass

    asyncio.run(main())


def test_limiter_releases_slot_when_waiter_is_cancelled():
    async def main():
        limiter = AsyncLimiter("test", max_concurrency=1)
        holding = asyncio.create_task(_tracked_work(limiter, {"active": 0, "peak": 0}, delay=0.1))
        await asyncio.sleep(0.01)  # let it take the slot

        waiter = asyncio.create_task(_tracked_work(limiter, {"active": 0, "peak": 0}))
        await asyncio.sleep(0.01)  # let it block on the semaphore
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        await holding
        async with asyncio.timeout(1):
            async with limiter.slot():
                pass

    asyncio.run(main())


def test_limiter_timeout_is_reported_and_does_not_leak():
    async def main():
        limiter = AsyncLimiter("test", max_concurrency=1)
        holding = asyncio.create_task(_tracked_work(limiter, {"active": 0, "peak": 0}, delay=0.2))
        await asyncio.sleep(0.01)

        with pytest.raises(LimitTimeout):
            await limiter.acquire(timeout=0.01)
        assert limiter.stats()["rejected"] == 1
        assert limiter.active == 1  # only the holder

        await holding
        assert limiter.active == 0

    asyncio.run(main())


def test_registry_is_one_limiter_per_loop_and_name():
    refs = []

    async def grab():
        first = limits.get("scraper")
        second = limits.get("scraper")
        assert first is second
        refs.append(first)
        return first

    asyncio.run(grab())
    asyncio.run(grab())
    # Two runs are two event loops: limiters must not be shared across them
    # (an asyncio.Semaphore is bound to the loop where it first blocks).
    assert refs[0] is not refs[1]


def test_registry_reads_environment_once_per_loop(monkeypatch):
    monkeypatch.setenv("SEARCH_CONCURRENCY", "5")

    async def main():
        limiter = limits.get("search:tavily")
        assert limiter.max_concurrency == 5
        snap = limits.snapshot()
        assert snap["search:tavily"]["max_concurrency"] == 5

    asyncio.run(main())


def test_token_bucket_spaces_starts():
    async def main():
        limiter = AsyncLimiter("test", rate_per_second=20, burst=1)  # 50ms per token
        started = time.perf_counter()
        for _ in range(4):
            async with limiter.slot():
                pass
        elapsed = time.perf_counter() - started
        # First is free, the next three wait ~50ms each.
        assert elapsed >= 0.12, elapsed

    asyncio.run(main())


def test_bounded_gather_caps_width_and_preserves_order():
    async def main():
        peak = {"active": 0, "peak": 0}

        async def job(i):
            peak["active"] += 1
            peak["peak"] = max(peak["peak"], peak["active"])
            try:
                await asyncio.sleep(0.01)
                return i
            finally:
                peak["active"] -= 1

        results = await bounded_gather([job(i) for i in range(12)], limit=4)
        assert results == list(range(12))
        assert peak["peak"] <= 4

    asyncio.run(main())


def test_bounded_gather_keeps_going_when_one_child_fails():
    async def main():
        async def good():
            await asyncio.sleep(0.01)
            return "ok"

        async def bad():
            raise ValueError("bad")

        results = await bounded_gather([good(), bad(), good()], limit=0)
        assert results[0] == "ok"
        assert isinstance(results[1], ValueError)
        assert results[2] == "ok"

    asyncio.run(main())


def test_bounded_gather_cancels_siblings_when_not_tolerating_errors():
    async def main():
        cancelled = {"count": 0}

        async def slow():
            try:
                await asyncio.sleep(5)
                return "late"
            except asyncio.CancelledError:
                cancelled["count"] += 1
                raise

        async def bad():
            await asyncio.sleep(0.01)
            raise ValueError("bad")

        with pytest.raises(ValueError):
            await asyncio.wait_for(
                bounded_gather([slow(), slow(), bad()], return_exceptions=False),
                timeout=2,
            )
        assert cancelled["count"] == 2

    asyncio.run(main())


def test_budget_truncates_and_counts_drops(monkeypatch):
    monkeypatch.setenv("MAX_URLS_PER_TASK", "4")
    monkeypatch.setenv("MAX_SUBQUERIES", "2")
    monkeypatch.setenv("MAX_SECTIONS", "1")

    with budget_scope() as budget:
        assert budget.allow_urls(list(range(10))) == [0, 1, 2, 3]
        assert budget.allow_urls([10, 11]) == []
        assert budget.urls_dropped == 8

        assert budget.allow_subqueries(["a", "b", "c"]) == ["a", "b"]
        assert budget.subqueries_dropped == 1

        assert budget.allow_sections(["s1", "s2"]) == ["s1"]
        assert budget.sections_dropped == 1

        report = budget.report()
        assert report == {"urls": 4, "subqueries": 2, "sections": 1, "dropped": 10}


def test_budget_scope_is_fresh_and_restores_parent():
    assert current_budget() is None
    with budget_scope(TaskBudget(max_urls=2)) as outer:
        outer.allow_urls(["a", "b"])
        assert outer.urls_started == 2

        with budget_scope() as inner:
            assert inner is not outer
            assert inner.urls_started == 0

        assert current_budget() is outer
    assert current_budget() is None


def test_budget_zero_means_unlimited():
    budget = TaskBudget(max_urls=0, max_subqueries=0, max_sections=0)
    assert budget.allow_urls(list(range(100))) == list(range(100))
    assert budget.dropped == 0
    assert budget.urls_started == 100


def test_worker_pools_share_one_process_wide_scraper_cap(monkeypatch):
    from gpt_researcher.utils.workers import WorkerPool

    monkeypatch.setenv("SCRAPER_GLOBAL_CONCURRENCY", "2")
    limits.reset()

    async def main():
        pool_a, pool_b = WorkerPool(5), WorkerPool(5)
        peak = {"active": 0, "peak": 0}

        async def fetch(pool):
            async with pool.throttle():
                peak["active"] += 1
                peak["peak"] = max(peak["peak"], peak["active"])
                try:
                    await asyncio.sleep(0.01)
                finally:
                    peak["active"] -= 1

        try:
            await asyncio.gather(*(fetch(pool_a if i % 2 else pool_b) for i in range(8)))
        finally:
            pool_a.executor.shutdown(wait=False)
            pool_b.executor.shutdown(wait=False)

        # Each pool allows 5, but the shared cap is 2. Before the shared
        # limiter this peaked at 8 (or at least above 2).
        assert peak["peak"] <= 2

    asyncio.run(main())


def test_scraper_drops_urls_beyond_task_budget():
    import requests

    from gpt_researcher.scraper.scraper import Scraper

    urls = [f"https://example.com/{i}" for i in range(10)]
    with budget_scope(TaskBudget(max_urls=3)):
        scraper = Scraper(urls, "test-agent", "bs", worker_pool=object())
        try:
            assert len(scraper.urls) == 3
            assert scraper.urls == urls[:3]
        finally:
            scraper.session.close()
    assert isinstance(scraper.session, requests.Session)


def test_search_cap_is_applied_per_provider(monkeypatch):
    from gpt_researcher.actions.query_processing import invoke_retriever_search

    monkeypatch.setenv("SEARCH_CONCURRENCY", "2")
    limits.reset()

    state = {"active": 0, "peak": 0}
    lock = threading.Lock()

    class FakeRetriever:
        def search(self, max_results=None):
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            try:
                time.sleep(0.02)
                return [{"href": "https://example.com"}]
            finally:
                with lock:
                    state["active"] -= 1

    async def main():
        await asyncio.gather(
            *(invoke_retriever_search(FakeRetriever(), None) for _ in range(8))
        )

    asyncio.run(main())
    assert state["peak"] <= 2


def test_llm_calls_share_one_process_wide_cap(monkeypatch):
    from gpt_researcher.utils import llm as llm_module

    monkeypatch.setenv("LLM_CONCURRENCY", "2")
    limits.reset()

    peak = {"active": 0, "peak": 0}

    class FakeProvider:
        last_response_metadata = None
        last_usage_metadata = None

        async def get_chat_response(self, messages, stream, websocket, **kwargs):
            peak["active"] += 1
            peak["peak"] = max(peak["peak"], peak["active"])
            try:
                await asyncio.sleep(0.02)
                return "ok"
            finally:
                peak["active"] -= 1

    monkeypatch.setattr(llm_module, "get_llm", lambda *a, **k: FakeProvider())

    async def main():
        await asyncio.gather(
            *(
                llm_module.create_chat_completion(
                    messages=[{"role": "user", "content": "hi"}],
                    model="fake-model",
                    llm_provider="fake",
                )
                for _ in range(6)
            )
        )

    asyncio.run(main())
    assert peak["peak"] <= 2


def test_summarizers_share_one_process_wide_cap(monkeypatch):
    from gpt_researcher.context.summarize import HierarchicalSummarizer

    monkeypatch.setenv("SUMMARY_CONCURRENCY", "2")
    limits.reset()

    peak = {"active": 0, "peak": 0}

    async def fake_llm(prompt, system):
        peak["active"] += 1
        peak["peak"] = max(peak["peak"], peak["active"])
        try:
            await asyncio.sleep(0.01)
            return "A factual note."
        finally:
            peak["active"] -= 1

    cfg = SimpleNamespace(smart_llm_model=None, context_token_budget=10_000)
    pages = [
        {"url": f"https://example.com/{i}", "title": "t", "raw_content": "Some source text. " * 20}
        for i in range(4)
    ]

    async def main():
        one = HierarchicalSummarizer(query="q", cfg=cfg, llm=fake_llm)
        two = HierarchicalSummarizer(query="q", cfg=cfg, llm=fake_llm)
        await asyncio.gather(one.async_get_context(pages), two.async_get_context(pages))

    asyncio.run(main())
    # Two instances x 4 pages = 8 calls; a per-instance cap of 2 would peak at 4.
    assert peak["peak"] <= 2
