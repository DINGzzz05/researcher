"""Runtime tool guard: step budget, duplicate calls and result clipping."""

from types import SimpleNamespace

import pytest

from gpt_researcher.actions.query_processing import invoke_retriever_search
from gpt_researcher.harness.guards import (
    RunGuard,
    current_guard,
    guard_from_config,
    guard_key,
    guard_scope,
)


class _FakeRetriever:
    """Counts invocations and returns one oversized result."""

    query = "battery cost"

    def __init__(self):
        self.calls = 0

    def search(self, **kwargs):
        self.calls += 1
        return [{"body": "x" * 50_000, "href": "https://example.com/a", "score": 1}]


def test_step_budget_refuses_calls_once_spent():
    guard = RunGuard(max_calls=2)

    assert guard.begin("p:q").allowed
    assert guard.begin("p:q2").allowed
    third = guard.begin("p:q3")

    assert not third.allowed
    assert "budget" in third.reason
    assert guard.calls == 2


def test_duplicate_call_is_served_from_cache():
    guard = RunGuard(max_calls=10, allow_repeat=0)

    first = guard.begin("p:q")
    guard.finish("p:q", [{"body": "first"}])
    second = guard.begin("p:q")

    assert first.allowed and not first.duplicate
    assert second.allowed and second.duplicate
    assert second.cached == [{"body": "first"}]
    assert guard.deduped == 1


def test_repeat_allowance_executes_then_dedupes():
    guard = RunGuard(max_calls=10, allow_repeat=1)

    assert not guard.begin("p:q").duplicate
    guard.finish("p:q", [{"body": "first"}])
    assert not guard.begin("p:q").duplicate  # one extra execution allowed
    guard.finish("p:q", [{"body": "second"}])
    assert guard.begin("p:q").duplicate


def test_cache_keeps_the_first_result_and_is_copied():
    guard = RunGuard()
    assert guard.begin("p:q").allowed
    guard.finish("p:q", [{"body": "first"}])

    cached = guard.begin("p:q").cached
    assert cached == [{"body": "first"}]
    cached[0]["body"] = "mutated"
    assert guard.begin("p:q").cached == [{"body": "first"}]


def test_truncation_clips_only_long_strings():
    guard = RunGuard(max_result_chars=10)

    results = guard.finish("p:q", [{"body": "abcdefghijklmnop", "score": 1, "ok": True}])

    assert results[0]["body"].startswith("abcdefghij")
    assert "truncated 6 chars" in results[0]["body"]
    assert results[0]["score"] == 1
    assert results[0]["ok"] is True
    assert guard.truncated == 1


def test_truncation_can_be_disabled():
    guard = RunGuard(max_result_chars=0)
    body = "x" * 1_000

    assert guard.finish("p:q", [{"body": body}])[0]["body"] == body


def test_guard_scope_is_isolated_and_reset():
    assert current_guard() is None
    with guard_scope(RunGuard(max_calls=1)) as guard:
        assert current_guard() is guard
    assert current_guard() is None


def test_guard_key_normalizes_whitespace_and_keeps_max_results():
    assert guard_key("TavilySearch", "  Battery   Cost ") == "tavilysearch:battery cost"
    assert guard_key("TavilySearch", "Battery Cost", 5) == "tavilysearch:battery cost:5"
    assert guard_key("TavilySearch", None) is None


def test_guard_from_config_off_when_every_limit_is_disabled():
    cfg = SimpleNamespace(max_tool_calls=0, max_duplicate_tool_calls=0, max_tool_result_chars=0)
    assert guard_from_config(cfg) is None


def test_guard_from_config_reads_the_limits():
    cfg = SimpleNamespace(max_tool_calls=7, max_duplicate_tool_calls=2, max_tool_result_chars=500)

    guard = guard_from_config(cfg)

    assert (guard.max_calls, guard.allow_repeat, guard.max_result_chars) == (7, 2, 500)


@pytest.mark.asyncio
async def test_duplicate_search_is_not_executed_twice():
    retriever = _FakeRetriever()
    guard = RunGuard(max_calls=10, max_result_chars=100)

    with guard_scope(guard):
        first = await invoke_retriever_search(retriever, None)
        second = await invoke_retriever_search(retriever, None)

    assert retriever.calls == 1
    assert guard.deduped == 1
    assert first == second
    assert "truncated" in first[0]["body"]


@pytest.mark.asyncio
async def test_exhausted_budget_skips_the_search():
    retriever = _FakeRetriever()
    guard = RunGuard(max_calls=1, max_result_chars=0)

    with guard_scope(guard):
        await invoke_retriever_search(retriever, None)
        skipped = await invoke_retriever_search(retriever, None)

    assert retriever.calls == 1
    assert skipped == []


@pytest.mark.asyncio
async def test_search_without_a_guard_is_unchanged():
    retriever = _FakeRetriever()

    results = await invoke_retriever_search(retriever, None)

    assert retriever.calls == 1
    assert len(results[0]["body"]) == 50_000
