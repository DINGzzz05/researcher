"""Rank fusion for multi-route context selection.

BM25 and embeddings fail differently: BM25 misses paraphrases and synonyms,
embeddings miss exact identifiers -- model numbers, tickers, rare surnames,
error codes -- because a single rare token carries little of the vector.
Running both and combining the *ranks* is more robust than picking one, and it
survives one route degrading instead of falling off a cliff.

Combining scores does not work: BM25 is unbounded, cosine similarity is in
[-1, 1], and their scales drift per sub-query. This module therefore combines
ranks with Reciprocal Rank Fusion (Cormack et al., 2009):

    score(d) = sum over routes of  weight / (k + rank(d))

``k`` flattens the curve so that the top few ranks do not dominate everything;
60 is the value from the original paper and needs no tuning per corpus.

The semantic route here ranks chunks by cosine similarity itself. The
``ContextCompressor`` used by ``CONTEXT_FILTER=embeddings`` only *filters* on a
similarity threshold and keeps page order (``EmbeddingsFilter`` preserves the
input order), which gives fusion nothing to fuse -- page order is not a
relevance rank.
"""

from __future__ import annotations

import asyncio
import math
import os
from typing import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .dedup import fingerprint
from .retriever import SearchAPIRetriever

#: RRF constant from the original paper; larger k flattens rank differences.
DEFAULT_RRF_K = 60
#: Chunks per route considered before fusion. Deeper than any final budget, so
#: a chunk that is mediocre on one route but strong on the other still wins.
DEFAULT_CANDIDATES = 60
#: Upper bound on chunks embedded per sub-query, to bound cost on huge pages.
MAX_EMBED_CHUNKS = 400


def chunk_key(document: Document) -> str:
    """Stable identity of a chunk, so the same text from two routes merges."""
    return fingerprint(getattr(document, "page_content", "") or "")


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Document]],
    *,
    k: int = DEFAULT_RRF_K,
    weights: Sequence[float] | None = None,
    limit: int | None = None,
) -> list[Document]:
    """Fuse several best-first document lists into one, by rank.

    Documents are identified by their normalized text, so a chunk returned by
    both routes is scored twice and appears once. Ties keep the order in which
    a document was first seen, which preserves the routes' own tie-breaking.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    weights = list(weights) if weights else [1.0] * len(ranked_lists)
    if len(weights) < len(ranked_lists):
        weights += [1.0] * (len(ranked_lists) - len(weights))

    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}
    first_seen: dict[str, int] = {}
    sequence = 0

    for list_index, ranked in enumerate(ranked_lists):
        weight = weights[list_index]
        seen_here: set[str] = set()
        for rank, document in enumerate(ranked or [], start=1):
            key = chunk_key(document)
            if not key or key in seen_here:
                continue
            seen_here.add(key)
            if key not in documents:
                documents[key] = document
                first_seen[key] = sequence
                sequence += 1
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)

    ordered = sorted(scores, key=lambda key: (-scores[key], first_seen[key]))
    if limit is not None:
        ordered = ordered[:limit]
    return [documents[key] for key in ordered]


def _chunks(documents: Sequence[dict], chunk_size: int, overlap: int) -> list[Document]:
    pages = SearchAPIRetriever(pages=list(documents)).invoke("")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return [c for c in splitter.split_documents(pages) if c.page_content.strip()]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def semantic_ranks(
    query: str,
    documents: Sequence[dict],
    embeddings,
    *,
    chunk_size: int = 1000,
    overlap: int = 100,
    max_chunks: int = MAX_EMBED_CHUNKS,
) -> list[Document]:
    """Chunks of ``documents``, best cosine similarity first.

    ``embeddings`` is any object with ``embed_documents``/``embed_query``
    (LangChain's ``Embeddings`` interface). Runs synchronously; callers wrap it
    in ``asyncio.to_thread``.
    """
    chunks = _chunks(documents, chunk_size, overlap)[:max_chunks]
    if not chunks:
        return []

    vectors = embeddings.embed_documents([c.page_content for c in chunks])
    query_vector = embeddings.embed_query(query)
    scored = [(_cosine(query_vector, vector), index) for index, vector in enumerate(vectors)]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [chunks[index] for _, index in scored]


async def async_semantic_ranks(
    query: str,
    documents: Sequence[dict],
    embeddings,
    **kwargs,
) -> list[Document]:
    """``semantic_ranks`` off the event loop, so concurrent sub-queries overlap."""
    return await asyncio.to_thread(semantic_ranks, query, documents, embeddings, **kwargs)


def hybrid_weights() -> tuple[float, float]:
    """(lexical, semantic) weights, from ``HYBRID_LEXICAL_WEIGHT``/``HYBRID_SEMANTIC_WEIGHT``."""
    try:
        lexical = float(os.environ.get("HYBRID_LEXICAL_WEIGHT", "1.0"))
        semantic = float(os.environ.get("HYBRID_SEMANTIC_WEIGHT", "1.0"))
    except ValueError:
        return 1.0, 1.0
    return lexical, semantic


def rrf_k() -> int:
    try:
        return max(0, int(os.environ.get("HYBRID_RRF_K", DEFAULT_RRF_K)))
    except ValueError:
        return DEFAULT_RRF_K


def candidate_limit(minimum: int) -> int:
    """Chunks per route to consider before fusion (``HYBRID_CANDIDATES``)."""
    try:
        configured = int(os.environ.get("HYBRID_CANDIDATES", DEFAULT_CANDIDATES))
    except ValueError:
        configured = DEFAULT_CANDIDATES
    return max(configured, minimum)
