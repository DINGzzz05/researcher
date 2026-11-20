"""Hierarchical summarization of scraped context.

Ranking keeps passages; summarization shrinks them. For a run over many long
pages, the writer does not need every paragraph - it needs the facts, numbers
and quotes that answer the question, attributed to their source. This module
builds that in three levels:

1. **chunk notes** - each part of a page that fits a model call is turned into
   dense factual notes,
2. **page summary** - the notes for one page are merged into one summary,
3. **context summary** - page summaries are merged (in batches) until the whole
   context fits the configured token budget.

Every level keeps ``Source:``/``Sources:`` lines, so the final context stays
traceable to the retrieved URLs. The module is used by ``select_context`` when
``CONTEXT_FILTER=summarize`` and degrades to keyword ranking if a call fails.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Awaitable, Callable, Sequence

from ..concurrency import limits
from ..concurrency.limiter import AsyncLimiter
from .budget import count_tokens, truncate_to_tokens

logger = logging.getLogger(__name__)

#: Prompt overhead reserved per call (question, source header, instructions).
_PROMPT_RESERVE_TOKENS = 400

_MAP_SYSTEM = (
    "You extract research notes. You answer only from the text you are given, "
    "never from prior knowledge. Reply with notes only: no preamble, no headings."
)
_COMBINE_SYSTEM = (
    "You merge research notes. Keep every fact, number, date and quote that "
    "helps answer the question, and keep source attributions. Reply with notes only."
)

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n{2,}")

#: (source URLs, rendered summary text). URLs are tracked separately from the
#: text so a merge across pages cannot lose which sources it came from.
Entry = tuple[tuple[str, ...], str]


class SummarizationError(RuntimeError):
    """The context could not be summarized (LLM failure or empty output)."""


def split_to_fit(text: str, max_tokens: int, model: str | None = None) -> list[str]:
    """Split ``text`` into parts of at most ``max_tokens`` tokens.

    Sentence boundaries are preferred; a single sentence longer than the budget
    is split on token boundaries. Empty text yields an empty list.
    """
    text = (text or "").strip()
    if not text:
        return []
    if max_tokens <= 0 or count_tokens(text, model) <= max_tokens:
        return [text]

    parts: list[str] = []
    buffer = ""
    for unit in (part.strip() for part in _SENTENCE_BREAK.split(text)):
        if not unit:
            continue
        candidate = f"{buffer} {unit}".strip() if buffer else unit
        if count_tokens(candidate, model) <= max_tokens:
            buffer = candidate
            continue
        if buffer:
            parts.append(buffer)
        if count_tokens(unit, model) <= max_tokens:
            buffer = unit
        else:
            parts.extend(_hard_split(unit, max_tokens, model))
            buffer = ""
    if buffer:
        parts.append(buffer)
    return parts or [text]


def _hard_split(text: str, max_tokens: int, model: str | None = None) -> list[str]:
    """Cut an oversized unit at token boundaries."""
    parts: list[str] = []
    remaining = text
    while remaining:
        chunk = truncate_to_tokens(remaining, max_tokens, model)
        if not chunk:
            break
        parts.append(chunk)
        remaining = remaining[len(chunk):].lstrip()
    return parts


class HierarchicalSummarizer:
    """Summarizes scraped pages into token-bounded, source-attributed context.

    ``cfg`` supplies the writer model (``smart_llm_model`` /
    ``smart_llm_provider``) and, when set, ``context_token_budget`` as the final
    size. ``llm`` is an optional ``async (prompt, system) -> str`` override used
    by tests and by callers with their own completion function.
    """

    def __init__(
        self,
        query: str,
        cfg,
        *,
        prompt_family=None,
        cost_callback: Callable[[float], None] | None = None,
        llm: Callable[[str, str], Awaitable[str]] | None = None,
        max_input_tokens: int | None = None,
        max_output_tokens: int | None = None,
        model: str | None = None,
        concurrency: int | None = None,
    ):
        self.query = query
        self.cfg = cfg
        self.prompt_family = prompt_family
        self.cost_callback = cost_callback
        self.llm = llm
        self.max_input_tokens = max(
            500, int(max_input_tokens or os.environ.get("SUMMARY_INPUT_TOKENS", "6000"))
        )
        self.max_output_tokens = int(
            max_output_tokens or os.environ.get("SUMMARY_OUTPUT_TOKENS", "700")
        )
        self.model = model or getattr(cfg, "smart_llm_model", None)
        final_budget = int(getattr(cfg, "context_token_budget", 0) or 0)
        self.final_budget = final_budget if final_budget > 0 else self.max_input_tokens
        # Shared across the process, not per summarizer: each sub-query builds
        # its own instance, so a per-instance cap multiplied by sections x
        # sub-queries (3 x 3 x 4 = 36 simultaneous 6k-token calls) and turned a
        # provider TPM limit into silent keyword fallback.
        if concurrency is not None:
            self._limiter = AsyncLimiter(
                "llm:summarize", max_concurrency=max(1, int(concurrency))
            )
        else:
            self._limiter = limits.get("llm:summarize")

    # -- LLM calls ---------------------------------------------------------

    async def _complete(self, prompt: str, system: str) -> str:
        async with self._limiter.slot():
            if self.llm is not None:
                response = await self.llm(prompt, system)
            else:
                from ..utils.llm import create_chat_completion

                response = await create_chat_completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=self.max_output_tokens,
                    llm_provider=getattr(self.cfg, "smart_llm_provider", None),
                    llm_kwargs=getattr(self.cfg, "llm_kwargs", None),
                    cost_callback=self.cost_callback,
                )

        if not isinstance(response, str) or not response.strip():
            raise SummarizationError("the model returned an empty summary")
        return response.strip()

    # -- prompts -----------------------------------------------------------

    def _map_prompt(self, url: str, title: str, text: str) -> str:
        return f"""Research question: {self.query}

Source: {url}
Title: {title}

Source text:
{text}

Write factual notes that help answer the question, using only the source text above.
- Keep exact numbers, dates, names and short quotes.
- Write in the language of the source text.
- No preamble and no markdown headings.
- If the text has nothing relevant, reply exactly: No relevant information.
"""

    def _combine_prompt(self, notes: Sequence[str], *, same_source: bool) -> str:
        note_budget = max(1, self.max_input_tokens - _PROMPT_RESERVE_TOKENS)
        notes = [truncate_to_tokens(note, note_budget) for note in notes]
        scope = "one source" if same_source else "several sources"
        attribution = (
            "Keep the source URL with each fact."
            if not same_source
            else "Keep the source URL."
        )
        return f"""Research question: {self.query}

Merge the notes from {scope} below into one dense summary.
- Keep every fact, number, date and quote that helps answer the question.
- {attribution}
- Do not add anything the notes do not say.
- Write in the language of the notes.
- No preamble and no markdown headings.

Notes:
{"\n\n".join(notes)}
"""

    # -- levels ------------------------------------------------------------

    async def _summarize_page(self, page: dict) -> str | None:
        url = page.get("source") or page.get("url") or ""
        title = page.get("title") or ""
        content = str(page.get("raw_content") or "")
        if not content.strip():
            return None

        parts = split_to_fit(
            content,
            max(1, self.max_input_tokens - _PROMPT_RESERVE_TOKENS),
            self.model,
        )
        if not parts:
            return None

        notes = await asyncio.gather(
            *[self._complete(self._map_prompt(url, title, part), _MAP_SYSTEM) for part in parts]
        )
        notes = [note for note in notes if note]
        if not notes:
            return None

        if len(parts) == 1:
            summary = notes[0]
        else:
            summary = await self._complete(
                self._combine_prompt(notes, same_source=True), _COMBINE_SYSTEM
            )

        if summary.strip().lower().startswith("no relevant information"):
            return None
        return summary

    def _batches(self, entries: Sequence[Entry]) -> list[list[Entry]]:
        """Pack entries into model-call-sized batches (at least two when possible)."""
        budget = max(1, self.max_input_tokens - _PROMPT_RESERVE_TOKENS)
        batches: list[list[Entry]] = []
        current: list[Entry] = []
        size = 0
        for entry in entries:
            cost = count_tokens(entry[1], self.model)
            if current and size + cost > budget:
                batches.append(current)
                current, size = [], 0
            current.append(entry)
            size += cost
        if current:
            batches.append(current)
        return batches

    async def async_get_context(self, pages: Sequence[dict] | None) -> str:
        """Summarized context for ``pages`` (dicts with ``raw_content``, ``url``, ``title``)."""
        pages = list(pages or [])
        if not pages:
            return ""

        results = await asyncio.gather(
            *[self._summarize_page(page) for page in pages],
            return_exceptions=True,
        )

        entries: list[Entry] = []
        for page, result in zip(pages, results):
            if isinstance(result, BaseException):
                logger.warning(
                    "Summarizing %s failed (%s); keeping the other pages",
                    page.get("url") or page.get("source") or "page",
                    result,
                )
                continue
            if not result:
                continue
            url = page.get("source") or page.get("url") or ""
            title = page.get("title") or ""
            entries.append(((url,), f"Source: {url}\nTitle: {title}\nSummary: {result}"))

        if not entries:
            raise SummarizationError("no page produced a summary")

        text = self._render(entries)
        while len(entries) > 1 and count_tokens(text, self.model) > self.final_budget:
            batches = self._batches(entries)
            if len(batches) <= 1:
                break

            merged: list[Entry] = []
            for batch in batches:
                sources = tuple(dict.fromkeys(url for entry in batch for url in entry[0]))
                combined = await self._complete(
                    self._combine_prompt([entry[1] for entry in batch], same_source=len(sources) == 1),
                    _COMBINE_SYSTEM,
                )
                header = (
                    f"Source: {sources[0]}"
                    if len(sources) == 1
                    else "Sources: " + ", ".join(sources)
                )
                merged.append((sources, f"{header}\nSummary: {combined}"))
            entries = merged
            text = self._render(entries)

        return truncate_to_tokens(text, self.final_budget, self.model)

    @staticmethod
    def _render(entries: Sequence[Entry]) -> str:
        return "\n\n".join(entry[1] for entry in entries)
