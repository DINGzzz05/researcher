"""Embedding storage and rank fusion for memory recall.

Memory recall used to be lexical only: BM25 over the stored text. That misses
paraphrases and cross-language matches, so each record now optionally carries
an embedding, and recall fuses two routes by rank:

    RRF(d) = sum over routes of 1 / (k + rank(d))

BM25 and embeddings fail differently - BM25 misses synonyms, embeddings miss
exact identifiers - and their scores are not comparable (BM25 unbounded,
cosine in [-1, 1]), so only the *ranks* are combined. This is the same fusion
the context pipeline uses for scraped chunks, applied to episodes and facts.

Vectors are stored as float32 blobs next to the record, tagged with the model
that produced them, so switching models invalidates old vectors instead of
silently mixing spaces.
"""

from __future__ import annotations

import array
import logging
import math
from typing import Sequence

logger = logging.getLogger(__name__)

#: RRF constant from the original paper; flattens rank differences.
RRF_K = 60


def pack(vector: Sequence[float]) -> bytes:
    """float32 blob for SQLite."""
    return array.array("f", (float(value) for value in vector)).tobytes()


def unpack(blob: bytes | None) -> tuple[float, ...]:
    """Inverse of :func:`pack`; ``()`` for missing or empty blobs."""
    if not blob:
        return ()
    values = array.array("f")
    values.frombytes(blob)
    return tuple(values)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity, 0 when either side is missing or dimensions differ."""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def fuse_ranks(
    ranked_lists: Sequence[Sequence[int]],
    *,
    k: int = RRF_K,
) -> dict[int, float]:
    """Reciprocal rank fusion over id lists (best first), keyed by id."""
    scores: dict[int, float] = {}
    for ranks in ranked_lists:
        seen: set[int] = set()
        for position, index in enumerate(ranks, start=1):
            if index in seen:
                continue
            seen.add(index)
            scores[index] = scores.get(index, 0.0) + 1.0 / (k + position)
    return scores


class LangchainEmbedder:
    """Adapter over the project's embedding providers.

    The client is built eagerly so a missing key or package disables vectors at
    configuration time, before a run pays for it; a later failure (quota, network)
    is handled by the caller, which falls back to BM25.
    """

    def __init__(self, provider: str, model: str, **kwargs):
        self.name = f"{provider}:{model}"
        self._provider = provider
        self._model = model
        self._kwargs = kwargs
        self._client = None

    def _ensure(self):
        if self._client is None:
            from .embeddings import Memory

            self._client = Memory(self._provider, self._model, **self._kwargs).get_embeddings()
        return self._client

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        client = self._ensure()
        return [list(vector) for vector in client.embed_documents(list(texts))]


def build_embedder(cfg):
    """Build the memory embedder from config, or ``None`` when disabled.

    ``MEMORY_EMBEDDING`` wins when set; empty means "reuse ``EMBEDDING``", the
    model the rest of the project already uses. ``off``/``none`` disables the
    vector route without touching the rest of the memory pipeline.
    """
    spec = (getattr(cfg, "memory_embedding", "") or "").strip()
    if spec.lower() in {"off", "none", "disabled", "false"}:
        return None
    if not spec:
        spec = (getattr(cfg, "embedding", "") or "").strip()

    provider, separator, model = spec.partition(":")
    if not separator or not provider or not model:
        return None

    try:
        embedder = LangchainEmbedder(provider.strip(), model.strip())
        embedder._ensure()
        return embedder
    except Exception as exc:  # missing package, missing key, bad model name
        logger.warning("Memory embeddings disabled (%s)", exc)
        return None


def sigmoid(value: float) -> float:
    """Map a reranker logit to (0, 1) without assuming its scale."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)
