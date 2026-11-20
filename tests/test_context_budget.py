"""Token counting and the context token budget."""

from types import SimpleNamespace

import pytest

from gpt_researcher.context.budget import (
    TokenBudget,
    count_tokens,
    fit_blocks,
    truncate_to_tokens,
)
from gpt_researcher.context.select import select_context


def _cfg(**overrides):
    cfg = SimpleNamespace(
        context_filter="keyword",
        context_dedup=True,
        context_dedup_threshold=0.9,
        context_token_budget=12000,
        similarity_threshold=0.42,
        smart_llm_model=None,
        smart_llm_provider=None,
        llm_kwargs={},
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _page(url="https://example.com/a", title="A", content="market growth 2024 " * 200):
    return {"url": url, "title": title, "raw_content": content}


def test_count_tokens_ignores_empty_text():
    assert count_tokens("") == 0
    assert count_tokens("   \n") == 0
    assert count_tokens(None) == 0


def test_count_tokens_grows_with_text():
    assert 0 < count_tokens("hello world") < count_tokens("hello world " * 50)


def test_truncate_to_tokens_keeps_a_prefix():
    text = "alpha beta gamma delta " * 40
    clipped = truncate_to_tokens(text, 10)
    assert count_tokens(clipped) <= 10
    assert text.startswith(clipped[:10])


def test_token_budget_is_unlimited_at_zero():
    budget = TokenBudget(0)
    assert budget.unlimited
    assert budget.take("anything at all") == "anything at all"


def test_fit_blocks_keeps_the_highest_ranked_passages():
    blocks = ["A" * 400, "B" * 400, "C" * 400]  # ~100 tokens each
    fitted = fit_blocks(blocks, 250)

    assert fitted.blocks == blocks[:2]
    assert fitted.truncated is True
    assert fitted.dropped == 1
    assert fitted.used_tokens <= 250


def test_fit_blocks_returns_everything_when_unlimited():
    blocks = ["one", "two", "three"]
    fitted = fit_blocks(blocks, 0)

    assert fitted.blocks == blocks
    assert fitted.truncated is False
    assert fitted.text == "\n".join(blocks)


def test_fit_blocks_truncates_a_first_passage_that_does_not_fit():
    fitted = fit_blocks(["x" * 10_000], 20)

    assert fitted.truncated is True
    assert fitted.blocks
    assert count_tokens(fitted.text) <= 20


@pytest.mark.asyncio
async def test_select_context_respects_the_token_budget(monkeypatch):
    monkeypatch.setenv("COMPRESSION_THRESHOLD", "0")
    cfg = _cfg(context_token_budget=60)

    out = await select_context("market growth", [_page()], cfg, max_results=10)

    assert out.startswith("Source: https://example.com/a")
    assert count_tokens(out) <= 60


@pytest.mark.asyncio
async def test_select_context_keeps_all_pages_when_budget_is_disabled(monkeypatch):
    monkeypatch.setenv("COMPRESSION_THRESHOLD", "0")
    cfg = _cfg(context_token_budget=0)

    out = await select_context(
        "market growth",
        [
            _page("https://example.com/a", "A"),
            _page("https://example.com/b", "B", "market growth battery costs fell " * 120),
        ],
        cfg,
        max_results=10,
    )

    assert out.count("Source:") == 2
