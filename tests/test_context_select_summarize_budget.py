"""The summarize filter is part of the same pipeline as the ranking filters.

Before this, ``CONTEXT_FILTER=summarize`` returned the summariser output
directly, so the token budget that bounds every other mode did not apply to it.
"""

from types import SimpleNamespace

import pytest

from gpt_researcher.context import select
from gpt_researcher.context.budget import count_tokens


class _FakeSummarizer:
    """Stands in for HierarchicalSummarizer: long text, no API calls."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def async_get_context(self, pages):
        return "summarized sentence. " * 4000


@pytest.mark.asyncio
async def test_summarize_mode_respects_the_token_budget(monkeypatch):
    import gpt_researcher.context.summarize as summarize_module

    monkeypatch.setattr(summarize_module, "HierarchicalSummarizer", _FakeSummarizer)

    cfg = SimpleNamespace(
        context_filter="summarize",
        context_dedup=True,
        context_dedup_threshold=0.9,
        context_token_budget=100,
        smart_llm_model=None,
    )
    pages = [{"url": "https://example.com/a", "title": "A", "raw_content": "x" * 20_000}]

    text = await select.select_context("battery cost", pages, cfg)

    assert text
    assert count_tokens(text) <= 100


@pytest.mark.asyncio
async def test_summarize_mode_still_shrinks_without_a_budget(monkeypatch):
    import gpt_researcher.context.summarize as summarize_module

    monkeypatch.setattr(summarize_module, "HierarchicalSummarizer", _FakeSummarizer)

    cfg = SimpleNamespace(
        context_filter="summarize",
        context_dedup=True,
        context_dedup_threshold=0.9,
        context_token_budget=0,
        smart_llm_model=None,
    )
    pages = [{"url": "https://example.com/a", "title": "A", "raw_content": "x" * 20_000}]

    text = await select.select_context("battery cost", pages, cfg)

    assert text.startswith("summarized sentence.")


@pytest.mark.asyncio
async def test_optional_summarize_stage_runs_after_ranking(monkeypatch):
    """CONTEXT_SUMMARIZE compresses whatever the ranking filter selected."""
    import gpt_researcher.context.summarize as summarize_module

    monkeypatch.setattr(summarize_module, "HierarchicalSummarizer", _FakeSummarizer)

    cfg = SimpleNamespace(
        context_filter="keyword",
        context_dedup=True,
        context_dedup_threshold=0.9,
        context_token_budget=200,
        context_summarize=True,
        smart_llm_model=None,
    )
    pages = [{"url": "https://example.com/a", "title": "A", "raw_content": "battery cost " * 1000}]

    text = await select.select_context("battery cost", pages, cfg)

    assert "summarized sentence." in text
    assert count_tokens(text) <= 200
