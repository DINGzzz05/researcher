"""Cross-encoder reranking for memory recall.

Fusion (BM25 + embeddings) decides *which* records are candidates; a
cross-encoder decides their final order. It sees query and candidate together,
so it catches what the retrieval routes cannot: a record that shares the words
but answers a different question, or one that answers the question without
sharing the words.

The model is optional: ``MEMORY_RERANK`` is off by default, and any failure
(missing package, model download, timeout) falls back to the fused scores, so a
run never pays for memory with a broken model.
"""

from __future__ import annotations

import logging
from typing import Sequence

from ..utils.rerank import load_scorer

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Local cross-encoder (sentence-transformers), loaded on first use."""

    def __init__(self, model_name: str):
        self.name = f"cross-encoder:{model_name}"
        self._model_name = model_name

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Relevance logit per candidate, aligned with ``texts``."""
        if not texts:
            return []
        scorer = load_scorer(self._model_name)
        return [float(value) for value in scorer([(query, text) for text in texts])]


def build_reranker(cfg):
    """Build the reranker from ``MEMORY_RERANK``, or ``None`` when off.

    Accepted forms: ``off`` (default), ``cross-encoder:<model>``, or a bare
    model name such as ``BAAI/bge-reranker-base``.
    """
    spec = (getattr(cfg, "memory_rerank", "") or "").strip()
    if not spec or spec.lower() in {"off", "none", "disabled", "false"}:
        return None

    model = spec.split(":", 1)[1].strip() if ":" in spec else spec
    if not model:
        return None

    try:
        import sentence_transformers  # noqa: F401
    except ImportError as exc:
        logger.warning("Memory rerank disabled (%s); install sentence-transformers", exc)
        return None
    return CrossEncoderReranker(model)
