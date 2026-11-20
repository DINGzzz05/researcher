"""Optional cross-encoder reranking for fused context candidates.

RRF only knows *ranks*: it cannot tell that two passages sharing a keyword are
about different things. A cross-encoder reads query and passage together, so it
separates those cases, at the price of a model download, memory, and ~50-200ms
per batch of candidates.

It is therefore opt-in. Enable it with config or the legacy environment
variables:

    export CONTEXT_RERANK=BAAI/bge-reranker-base
    pip install sentence-transformers

``RERANK_MODEL`` / ``RERANK_TOP_N`` still work when the config keys are unset.
Without a model name the function is the identity, so nothing changes for
existing setups and no dependency is imported. Tests and callers that already
own a scorer can inject one through ``scorer``.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Sequence

from langchain_core.documents import Document

from ..utils.rerank import order_by_scores, score_pairs

logger = logging.getLogger(__name__)

#: Rerank at most this many fused candidates (cost grows linearly).
DEFAULT_TOP_N = 30

_DISABLED = {"off", "none", "disabled", "false"}


def context_rerank_model(cfg) -> str | None:
    """Cross-encoder name from ``CONTEXT_RERANK`` (or legacy ``RERANK_MODEL``)."""
    spec = (getattr(cfg, "context_rerank", "") or "").strip()
    if not spec:
        spec = (os.environ.get("RERANK_MODEL") or "").strip()
    if not spec or spec.lower() in _DISABLED:
        return None
    return spec.split(":", 1)[1].strip() if ":" in spec else spec


def context_rerank_top_n(cfg) -> int | None:
    """``CONTEXT_RERANK_TOP_N`` candidates to score; 0 disables reranking."""
    setting = getattr(cfg, "context_rerank_top_n", None)
    if setting is None:
        setting = os.environ.get("RERANK_TOP_N", DEFAULT_TOP_N)
    try:
        value = int(setting)
    except (TypeError, ValueError):
        value = DEFAULT_TOP_N
    return None if value <= 0 else value


def rerank_documents(
    query: str,
    documents: Sequence[Document],
    *,
    model: str | None = None,
    top_n: int | None = None,
    scorer: Callable[[Sequence[tuple[str, str]]], Sequence[float]] | None = None,
) -> list[Document]:
    """Reordered documents, best first. Identity when no model or scorer is given.

    A model that cannot be loaded (missing package, no weights, offline) is
    logged and skipped: reranking improves ordering, it must never fail a run.
    """
    docs = list(documents or [])
    if not docs:
        return []

    scores = score_pairs(
        query,
        [doc.page_content or "" for doc in docs],
        model=model,
        scorer=scorer,
        label="context rerank",
    )
    if scores is None:
        return docs[:top_n] if top_n else docs

    ranked = [docs[index] for index in order_by_scores(scores)]
    return ranked[:top_n] if top_n else ranked
