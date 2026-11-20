"""Cross-encoder scoring shared by context retrieval and memory recall.

Both pipelines retrieve more candidates than they can afford to hand to the
model, then want a scorer that reads query and text *together* to reorder the
top of the list. Loading, caching, error handling and turning scores into an
order are identical; what differs is which candidates are scored, how many, and
what the caller does with the result - and that stays with the caller.

The dependency is optional: nothing here imports ``sentence-transformers``
until a model is actually requested, and a model that cannot be loaded degrades
to "no scores" rather than an exception, so reranking never fails a run.
"""

from __future__ import annotations

import logging
from typing import Callable, Sequence

logger = logging.getLogger(__name__)

#: ``(query, text)`` pairs in, one score per pair out.
Scorer = Callable[[Sequence[tuple[str, str]]], Sequence[float]]

_MODEL_CACHE: dict[str, Scorer] = {}


def load_scorer(model: str) -> Scorer:
    """Cached cross-encoder for ``model``, built on first use."""
    cached = _MODEL_CACHE.get(model)
    if cached is not None:
        return cached

    from sentence_transformers import CrossEncoder  # optional dependency

    encoder = CrossEncoder(model)

    def score(pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        return encoder.predict(list(pairs), show_progress_bar=False)

    _MODEL_CACHE[model] = score
    return score


def score_pairs(
    query: str,
    texts: Sequence[str],
    *,
    model: str | None = None,
    scorer: Scorer | None = None,
    label: str = "rerank",
) -> list[float] | None:
    """One score per text, or ``None`` when no usable scorer is available."""
    if not texts:
        return []

    if scorer is None:
        if not model:
            return None
        try:
            scorer = load_scorer(model)
        except Exception as exc:  # missing package, missing weights, offline
            logger.warning("%s model %s unavailable (%s); keeping previous order", label, model, exc)
            return None

    try:
        scores = [float(value) for value in scorer([(query, text) for text in texts])]
    except Exception as exc:  # noqa: BLE001 - ordering must not break the run
        logger.warning("%s failed (%s); keeping previous order", label, exc)
        return None

    if len(scores) != len(texts):
        logger.warning(
            "%s returned %s scores for %s texts; keeping previous order",
            label, len(scores), len(texts),
        )
        return None
    return scores


def order_by_scores(scores: Sequence[float]) -> list[int]:
    """Indices sorted by score, best first, ties keeping the input order."""
    return sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
