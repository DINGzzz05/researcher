"""Rank fusion, semantic ranking and optional reranking."""

import pytest
from langchain_core.documents import Document

from gpt_researcher.context import select as select_module
from gpt_researcher.context.fusion import (
    chunk_key,
    reciprocal_rank_fusion,
    semantic_ranks,
)
from gpt_researcher.context.rerank import rerank_documents


def doc(text: str) -> Document:
    return Document(page_content=text, metadata={"source": "https://example.com"})


class BagOfWordsEmbeddings:
    """Deterministic vectors: one dimension per tracked term."""

    def __init__(self, terms):
        self.terms = list(terms)

    def _vector(self, text: str):
        low = text.lower()
        return [float(low.count(term)) for term in self.terms]

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


# -- reciprocal rank fusion -------------------------------------------------


def test_same_rank_on_both_routes_beats_a_single_route_hit():
    # L5 is fifth on both routes (2 / 65); L1 is first on the lexical route
    # only (1 / 61). Two moderate votes beat one strong one.
    left = [doc(f"L{index}") for index in range(1, 7)]
    right = [doc(f"R{index}") for index in range(1, 5)] + [doc("L5")]

    fused = reciprocal_rank_fusion([left, right])

    assert fused[0].page_content == "L5"


def test_fusion_merges_the_same_chunk_instead_of_duplicating_it():
    fused = reciprocal_rank_fusion([[doc("same text")], [doc("same  text")]])

    assert len(fused) == 1
    assert chunk_key(doc("same text")) == chunk_key(doc("same  text"))


def test_scores_fall_with_rank_and_k_flattens_the_curve():
    lists = [[doc("first"), doc("second")]]

    sharp = reciprocal_rank_fusion(lists, k=0)
    flat = reciprocal_rank_fusion(lists, k=60)

    assert [d.page_content for d in sharp] == [d.page_content for d in flat]
    # A larger k compresses the gap between adjacent ranks.
    sharp_gap = 1 / 1 - 1 / 2
    flat_gap = 1 / 61 - 1 / 62
    assert flat_gap < sharp_gap


def test_weights_shift_the_balance():
    left = [doc("lexical")]
    right = [doc("semantic")]

    lexical_only = reciprocal_rank_fusion([left, right], weights=[10.0, 1.0])

    assert lexical_only[0].page_content == "lexical"


def test_limit_and_empty_input():
    assert len(reciprocal_rank_fusion([[doc("a"), doc("b")]], limit=1)) == 1
    assert reciprocal_rank_fusion([]) == []


# -- semantic route ---------------------------------------------------------


def test_semantic_ranks_order_by_cosine_similarity():
    embeddings = BagOfWordsEmbeddings(["solar", "battery"])
    pages = [
        {"raw_content": "Battery pack prices fell again this quarter.", "url": "u1"},
        {"raw_content": "Solar cell efficiency improved in the lab.", "url": "u2"},
    ]

    ranked = semantic_ranks("solar efficiency", pages, embeddings)

    assert ranked[0].page_content.startswith("Solar cell")


# -- hybrid selection -------------------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_select_prefers_the_chunk_both_routes_like(monkeypatch):
    pages = [{"raw_content": "chunk one", "url": "u"}]
    lexical = [doc("A"), doc("B"), doc("C"), doc("D"), doc("E")]
    semantic = [doc("S1"), doc("S2"), doc("S3"), doc("S4"), doc("B")]

    monkeypatch.setattr(select_module, "LexicalContextCompressor", lambda **_: type(
        "Lex", (), {"select": lambda self, query, limit: lexical}
    )())

    async def fake_semantic(query, documents, embeddings, **kwargs):
        return semantic

    monkeypatch.setattr(select_module, "async_semantic_ranks", fake_semantic)

    docs = await select_module.hybrid_select(
        "q", pages, None, max_results=3, embeddings=lambda: object()
    )

    # B is fifth on both routes; nothing else appears on both.
    assert docs[0].page_content == "B"


@pytest.mark.asyncio
async def test_hybrid_select_keeps_bm25_ranking_when_embeddings_fail(monkeypatch):
    lexical = [doc("A"), doc("B")]

    monkeypatch.setattr(select_module, "LexicalContextCompressor", lambda **_: type(
        "Lex", (), {"select": lambda self, query, limit: lexical}
    )())

    def broken_embeddings():
        raise RuntimeError("no embedding provider")

    docs = await select_module.hybrid_select("q", [{"raw_content": "x"}], None, embeddings=broken_embeddings)

    assert [d.page_content for d in docs] == ["A", "B"]


@pytest.mark.asyncio
async def test_hybrid_select_without_embeddings_is_keyword_ranking(monkeypatch):
    lexical = [doc("A"), doc("B")]

    monkeypatch.setattr(select_module, "LexicalContextCompressor", lambda **_: type(
        "Lex", (), {"select": lambda self, query, limit: lexical}
    )())

    docs = await select_module.hybrid_select("q", [{"raw_content": "x"}], None)

    assert [d.page_content for d in docs] == ["A", "B"]


# -- reranking --------------------------------------------------------------


def test_rerank_is_the_identity_without_a_model(monkeypatch):
    monkeypatch.delenv("RERANK_MODEL", raising=False)
    docs = [doc("A"), doc("B")]

    assert rerank_documents("q", docs) == docs


def test_rerank_uses_an_injected_cross_encoder():
    docs = [doc("A"), doc("B"), doc("C")]

    ranked = rerank_documents("q", docs, scorer=lambda pairs: [0.1, 0.9, 0.5])

    assert [d.page_content for d in ranked] == ["B", "C", "A"]


def test_rerank_keeps_order_when_scores_do_not_line_up():
    docs = [doc("A"), doc("B")]

    assert rerank_documents("q", docs, scorer=lambda pairs: [0.5]) == docs
