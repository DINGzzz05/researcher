"""Token budgets for the context a research run sends to the writer.

The context filter (``select.py``) decides *which* passages are relevant; this
module decides how many of them fit. A relevance filter keeps the best chunks
but has no opinion about their total size, so a run over many long pages can
still overflow the writer's window and pay for text the model never needed.

Counting uses tiktoken for the writer's model when available and falls back to
a ~4-characters-per-token estimate, so a run never fails because an encoding
cannot be fetched. The budget is applied *after* ranking, so the passages kept
are always the highest-ranked ones that fit.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)

#: tiktoken encoding used when the target model is unknown to tiktoken.
DEFAULT_ENCODING = "cl100k_base"
#: English fallback ratio: one token per four characters.
CHARS_PER_TOKEN = 4
#: Separator ``PromptFamily.pretty_print_docs`` puts between passages.
BLOCK_SEPARATOR = "\n"

# Model name -> encoder, or None when tiktoken could not provide one. Cache both
# outcomes: a missing tiktoken install should not be retried per chunk.
_ENCODERS: dict[str | None, object | None] = {}


def _encoder(model: str | None):
    """Cached tiktoken encoder for ``model``; ``None`` when tiktoken is unusable."""
    if model in _ENCODERS:
        return _ENCODERS[model]

    encoder = None
    try:
        import tiktoken

        try:
            # GPT-5.x and other newest names are often unknown to the pinned
            # tiktoken, so fall back to the encoding every modern model shares.
            encoder = tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding(DEFAULT_ENCODING)
        except Exception:
            encoder = tiktoken.get_encoding(DEFAULT_ENCODING)
    except Exception as exc:  # missing package or encoding data that cannot be downloaded
        logger.debug("tiktoken unavailable (%s); estimating tokens from characters", exc)

    _ENCODERS[model] = encoder
    return encoder


def count_tokens(text: str | None, model: str | None = None) -> int:
    """Number of tokens ``text`` occupies. Empty and whitespace-only text is 0."""
    if not text:
        return 0
    text = str(text)
    if not text.strip():
        return 0

    encoder = _encoder(model)
    if encoder is not None:
        try:
            return len(encoder.encode(text, disallowed_special=()))
        except Exception as exc:  # e.g. text that trips a tiktoken edge case
            logger.debug("tiktoken encode failed (%s); estimating from characters", exc)
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


def truncate_to_tokens(text: str, max_tokens: int, model: str | None = None) -> str:
    """``text`` cut to at most ``max_tokens`` tokens."""
    if not text or max_tokens <= 0:
        return ""
    text = str(text)
    if count_tokens(text, model) <= max_tokens:
        return text

    encoder = _encoder(model)
    if encoder is not None:
        try:
            tokens = encoder.encode(text, disallowed_special=())
            # A token boundary can split a multi-byte character; drop the
            # replacement character decode leaves behind.
            return encoder.decode(tokens[:max_tokens]).rstrip("\ufffd")
        except Exception as exc:
            logger.debug("tiktoken truncate failed (%s); cutting by characters", exc)
    return text[: max_tokens * CHARS_PER_TOKEN]


class TokenBudget:
    """A running token allowance. ``limit`` of ``None`` or ``0`` means unlimited."""

    def __init__(self, limit: int | None = None, model: str | None = None):
        self.limit = int(limit) if limit and int(limit) > 0 else None
        self.model = model
        self.used = 0

    @property
    def unlimited(self) -> bool:
        return self.limit is None

    @property
    def remaining(self) -> int | None:
        if self.limit is None:
            return None
        return max(0, self.limit - self.used)

    def fits(self, text: str | None) -> bool:
        """Whether ``text`` still fits in the remaining allowance."""
        if self.limit is None:
            return True
        return count_tokens(text, self.model) <= (self.remaining or 0)

    def add(self, text: str | None) -> int:
        """Account for ``text`` and return its token count."""
        tokens = count_tokens(text, self.model)
        self.used += tokens
        return tokens

    def take(self, text: str | None) -> str:
        """``text`` if it fits, else the longest prefix that does (possibly "")."""
        if not text:
            return ""
        if self.limit is None:
            self.add(text)
            return str(text)
        remaining = self.remaining or 0
        if remaining <= 0:
            return ""
        clipped = truncate_to_tokens(str(text), remaining, self.model)
        self.add(clipped)
        return clipped


@dataclass
class FitResult:
    """Outcome of fitting formatted passages to a budget."""

    text: str
    blocks: list[str] = field(default_factory=list)
    used_tokens: int = 0
    truncated: bool = False
    dropped: int = 0


def fit_blocks(
    blocks: Sequence[str],
    limit: int | None,
    model: str | None = None,
    separator: str = BLOCK_SEPARATOR,
) -> FitResult:
    """Keep ``blocks`` in order until ``limit`` tokens are used.

    When the first passage alone exceeds the budget, a prefix of it is kept so
    the caller still has context to work with. Later passages are never cut:
    a half passage ranks below a whole, higher-ranked one.
    """
    budget = TokenBudget(limit, model)
    kept: list[str] = []
    truncated = False
    dropped = 0

    for index, block in enumerate(blocks):
        candidate = (separator if kept else "") + block
        if budget.fits(candidate):
            budget.add(candidate)
            kept.append(block)
            continue

        if not kept:
            clipped = budget.take(candidate)
            if clipped.strip():
                kept.append(clipped)
            truncated = True
        else:
            truncated = True
            dropped = len(blocks) - index
        break

    return FitResult(
        text=separator.join(kept),
        blocks=kept,
        used_tokens=budget.used,
        truncated=truncated,
        dropped=dropped,
    )


def fit_text(text: str, limit: int | None, model: str | None = None) -> str:
    """``text`` shortened to ``limit`` tokens (unlimited when ``limit`` is falsy)."""
    if not limit or int(limit) <= 0:
        return text
    return truncate_to_tokens(text, int(limit), model)


def total_tokens(items: Iterable[str], model: str | None = None) -> int:
    """Sum of ``count_tokens`` over ``items``."""
    return sum(count_tokens(item, model) for item in items)
