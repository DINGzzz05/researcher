"""Hierarchical summarization of context."""

from types import SimpleNamespace

import pytest

from gpt_researcher.context.budget import count_tokens
from gpt_researcher.context.select import select_context
from gpt_researcher.context.summarize import (
    _COMBINE_SYSTEM,
    _MAP_SYSTEM,
    HierarchicalSummarizer,
    SummarizationError,
    split_to_fit,
)


class FakeLLM:
    """Async stand-in for the completion call, with distinguishable phases."""

    def __init__(self, map_response: str = "NOTE", combine_response: str = "COMBINED"):
        self.map_response = map_response
        self.combine_response = combine_response
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, prompt: str, system: str) -> str:
        self.calls.append((prompt, system))
        if system == _COMBINE_SYSTEM:
            return self.combine_response
        assert system == _MAP_SYSTEM
        return self.map_response


def _cfg(**overrides):
    cfg = SimpleNamespace(
        smart_llm_model=None,
        smart_llm_provider=None,
        llm_kwargs={},
        context_token_budget=0,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _page(url: str = "https://example.com/a", content: str | None = None, title: str = "A") -> dict:
    return {
        "url": url,
        "title": title,
        "raw_content": content if content is not None else "The market grew twelve percent in 2024. " * 40,
    }


def test_split_to_fit_keeps_each_part_within_budget():
    text = "The market grew twelve percent in 2024. " * 60

    parts = split_to_fit(text, 100)

    assert len(parts) > 1
    assert all(count_tokens(part) <= 100 for part in parts)
    assert "".join("".join(part.split()) for part in parts) == "".join(text.split())


@pytest.mark.asyncio
async def test_summarizer_combines_chunks_then_keeps_the_source():
    llm = FakeLLM()
    summarizer = HierarchicalSummarizer(
        "how fast did the market grow", _cfg(), llm=llm, max_input_tokens=600, max_output_tokens=50
    )

    out = await summarizer.async_get_context([_page()])

    assert out.startswith("Source: https://example.com/a")
    assert "Summary: COMBINED" in out
    # chunk notes were merged before the page summary was emitted
    assert any(system == _COMBINE_SYSTEM for _, system in llm.calls)
    assert any(system == _MAP_SYSTEM for _, system in llm.calls)


@pytest.mark.asyncio
async def test_summarizer_merges_across_pages_until_the_budget_fits():
    llm = FakeLLM(map_response="X" * 2000, combine_response="MERGED")
    cfg = _cfg(context_token_budget=200)
    summarizer = HierarchicalSummarizer(
        "q", cfg, llm=llm, max_input_tokens=1000, max_output_tokens=50
    )
    pages = [_page(f"https://example.com/{i}", title=f"T{i}", content=f"fact number {i}") for i in range(4)]

    out = await summarizer.async_get_context(pages)

    assert "MERGED" in out
    assert out.count("MERGED") == 2  # two batches were merged
    assert "Sources:" in out
    assert all(f"https://example.com/{i}" in out for i in range(4))
    assert count_tokens(out) <= 200


@pytest.mark.asyncio
async def test_summarizer_raises_when_every_page_fails():
    async def broken(prompt: str, system: str) -> str:
        raise RuntimeError("provider down")

    summarizer = HierarchicalSummarizer("q", _cfg(), llm=broken, max_input_tokens=600)

    with pytest.raises(SummarizationError):
        await summarizer.async_get_context([_page()])


@pytest.mark.asyncio
async def test_select_context_uses_the_summarize_mode(monkeypatch):
    import gpt_researcher.context.summarize as summarize_module

    async def fake_summary(self, pages):
        assert pages
        return "SUMMARY CONTEXT"

    monkeypatch.setattr(summarize_module.HierarchicalSummarizer, "async_get_context", fake_summary)
    monkeypatch.setenv("COMPRESSION_THRESHOLD", "0")
    cfg = _cfg(context_filter="summarize", context_dedup=True, context_dedup_threshold=0.9)

    out = await select_context("q", [_page()], cfg)

    assert out == "SUMMARY CONTEXT"


@pytest.mark.asyncio
async def test_select_context_falls_back_to_keyword_when_summarizing_fails(monkeypatch):
    import gpt_researcher.context.summarize as summarize_module

    async def broken(self, pages):
        raise SummarizationError("no summary")

    monkeypatch.setattr(summarize_module.HierarchicalSummarizer, "async_get_context", broken)
    monkeypatch.setenv("COMPRESSION_THRESHOLD", "0")
    cfg = _cfg(context_filter="summarize", context_dedup=True, context_dedup_threshold=0.9)

    out = await select_context("q", [_page()], cfg)

    assert out.startswith("Source: https://example.com/a")
