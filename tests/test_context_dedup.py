"""Page and chunk deduplication for context selection."""

from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from gpt_researcher.context.dedup import (
    dedupe_documents,
    dedupe_pages,
    fingerprint,
    jaccard,
    normalize_url,
    shingles,
)
from gpt_researcher.context.select import select_context

_BASE = (
    "The global battery market grew twelve percent in 2024 as electric vehicle "
    "demand increased across Europe and China. Manufacturers added capacity in "
    "Hungary and Poland, while lithium prices fell for the third consecutive quarter. "
    "Analysts expect further cost declines through 2026 as recycling capacity expands."
)


def _doc(text: str, url: str = "https://example.com/a") -> Document:
    return Document(page_content=text, metadata={"source": url, "title": "Doc"})


def _cfg(**overrides):
    cfg = SimpleNamespace(
        context_filter="keyword",
        context_dedup=True,
        context_dedup_threshold=0.9,
        context_token_budget=0,
        similarity_threshold=0.42,
        smart_llm_model=None,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_fingerprint_ignores_case_and_whitespace():
    assert fingerprint("Hello   World") == fingerprint("hello world")


def test_jaccard_matches_identical_text_and_rejects_unrelated_text():
    assert jaccard(shingles(_BASE), shingles(_BASE)) == 1.0
    assert jaccard(shingles(_BASE), shingles("completely different words about cooking")) == 0.0


def test_dedupe_documents_removes_exact_duplicates():
    documents = [_doc(_BASE, "https://example.com/a"), _doc(_BASE, "https://example.com/b")]

    assert len(dedupe_documents(documents)) == 1


def test_dedupe_documents_removes_near_duplicates():
    near = _BASE + " The report was published on a Tuesday afternoon."
    documents = [_doc(_BASE, "https://example.com/a"), _doc(near, "https://example.com/b")]

    assert len(dedupe_documents(documents, threshold=0.8)) == 1
    # At the exact-duplicate-only setting both are kept.
    assert len(dedupe_documents(documents, threshold=1.0)) == 2


def test_dedupe_documents_keeps_distinct_chunks():
    documents = [
        _doc(_BASE, "https://example.com/a"),
        _doc("A recipe for sourdough bread with rye flour and a long cold ferment.", "https://example.com/b"),
    ]

    assert len(dedupe_documents(documents)) == 2


def test_dedupe_pages_drops_repeated_url_and_repeated_content():
    pages = [
        {"url": "https://example.com/a", "raw_content": _BASE},
        {"url": "https://example.com/a", "raw_content": "different body"},
        {"url": "https://example.com/b", "raw_content": _BASE},
        {"url": "https://example.com/c", "raw_content": "unrelated content"},
    ]

    kept = dedupe_pages(pages)

    assert [page["url"] for page in kept] == ["https://example.com/a", "https://example.com/c"]


def test_normalize_url_strips_fragment_and_trailing_slash():
    assert normalize_url("https://Example.com/a/#section") == "https://example.com/a"


@pytest.mark.asyncio
async def test_select_context_dedupes_identical_pages(monkeypatch):
    monkeypatch.setenv("COMPRESSION_THRESHOLD", "0")
    cfg = _cfg()

    out = await select_context(
        "battery market growth",
        [
            {"url": "https://example.com/a", "title": "A", "raw_content": _BASE},
            {"url": "https://example.com/b", "title": "B", "raw_content": _BASE},
        ],
        cfg,
        max_results=10,
    )

    assert out.count("Source:") == 1
