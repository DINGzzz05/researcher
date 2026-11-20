"""Turns pages of text into LLM context, the same way everywhere.

Research sub-queries (scraped pages) and report chat (the finished report)
both call ``select_context``. ``CONTEXT_FILTER`` picks how pages become context:

  auto        jev when TYPESAFE_API_KEY is set, otherwise keyword
  jev         TypeSafe's Jev scores each chunk's usefulness (needs a key)
  hybrid      BM25 and embedding routes, fused by reciprocal rank (needs embeddings)
  keyword     BM25 keyword ranking: local, no API, no model, no embeddings
  embeddings  embedding similarity (needs an embedding provider)
  summarize   hierarchical LLM summary: chunk notes -> page -> whole context
  none        no filtering: every page, in full

Whatever the mode, the same pipeline runs before the context is rendered:

1. **dedup** - identical pages and exact/near-duplicate chunks are dropped
   (``CONTEXT_DEDUP``, ``CONTEXT_DEDUP_THRESHOLD``),
2. **summarize** - optional: ``CONTEXT_SUMMARIZE`` compresses the selected
   chunks into one summary before the budget is applied,
3. **token budget** - the highest-ranked passages that fit
   ``CONTEXT_TOKEN_BUDGET`` are kept (0 disables the cap).

Anything that fails degrades to ``keyword``, so no mode can leave a run
without context. Benchmarks: ``evals/context_filter``.
"""

import logging
import os
from typing import Callable

from langchain_core.documents import Document

from ..prompts import PromptFamily
from .budget import fit_blocks, fit_text
from .dedup import dedupe_documents, dedupe_pages
from .fusion import (
    async_semantic_ranks,
    candidate_limit,
    hybrid_weights,
    reciprocal_rank_fusion,
    rrf_k,
)
from .jev_filter import JevContextCompressor, JevError
from .lexical import LexicalContextCompressor
from .rerank import context_rerank_model, context_rerank_top_n, rerank_documents
from .retriever import SearchAPIRetriever

logger = logging.getLogger(__name__)

CONTEXT_FILTERS = ("auto", "jev", "hybrid", "keyword", "embeddings", "summarize", "none")

#: Keyword chunks must score at least this fraction of the best chunk, and
#: up to this many are kept. Chosen by evals/context_filter: the threshold
#: is what beats embeddings; plain top-10 BM25 loses to them.
KEYWORD_RELATIVE_THRESHOLD = float(os.environ.get("KEYWORD_RELATIVE_THRESHOLD", "0.5"))
KEYWORD_MAX_RESULTS = int(os.environ.get("KEYWORD_MAX_RESULTS", "25"))

#: Fallbacks for callers that pass a duck-typed config instead of ``Config``.
DEFAULT_CONTEXT_TOKEN_BUDGET = int(os.environ.get("CONTEXT_TOKEN_BUDGET", "12000"))
DEFAULT_DEDUP_THRESHOLD = 0.9


def resolve_context_filter(setting: str | None) -> str:
    """The filter to use; ``auto`` means Jev when a TypeSafe key is set, else keyword."""
    mode = (setting or "auto").strip().lower()
    if mode not in CONTEXT_FILTERS:
        logger.warning(f"Unknown CONTEXT_FILTER {setting!r}; expected one of {CONTEXT_FILTERS}. Using 'auto'.")
        mode = "auto"
    if mode == "auto":
        return "jev" if os.environ.get("TYPESAFE_API_KEY") else "keyword"
    return mode


def _as_documents(pages: list[dict]) -> list[Document]:
    return SearchAPIRetriever(pages=pages).invoke("")


def _dedup_enabled(cfg) -> bool:
    return bool(getattr(cfg, "context_dedup", True))


def _summarize_enabled(cfg) -> bool:
    return bool(getattr(cfg, "context_summarize", False))


def _dedup_threshold(cfg) -> float:
    value = getattr(cfg, "context_dedup_threshold", None)
    return DEFAULT_DEDUP_THRESHOLD if value is None else float(value)


def _token_budget(cfg) -> int:
    value = getattr(cfg, "context_token_budget", None)
    if value is None:
        return DEFAULT_CONTEXT_TOKEN_BUDGET
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return DEFAULT_CONTEXT_TOKEN_BUDGET


CONTEXT_ORDERS = ("relevance", "lost_in_middle")


def _context_order(cfg) -> str:
    """How selected passages are presented: rank order, or edges-first."""
    mode = (getattr(cfg, "context_order", "") or os.environ.get("CONTEXT_ORDER", "relevance"))
    mode = str(mode).strip().lower()
    if mode not in CONTEXT_ORDERS:
        logger.warning(
            "Unknown CONTEXT_ORDER %r; expected one of %s. Using 'relevance'.",
            mode, CONTEXT_ORDERS,
        )
        return "relevance"
    return mode


def _order_blocks(blocks: list[str], cfg) -> list[str]:
    """Best passages at both ends, weakest in the middle.

    Long contexts are read unevenly: material at the start and end gets more
    attention than material buried in the middle (Liu et al., "Lost in the
    Middle"). This runs *after* the budget so reordering cannot push the
    second-best passage past the cut.
    """
    if _context_order(cfg) != "lost_in_middle" or len(blocks) < 3:
        return blocks
    return blocks[::2] + blocks[1::2][::-1]


def _render(
    docs: list[Document], top_n: int | None, cfg, prompt_family, *, ranked: bool = True
) -> str:
    """Format ranked documents, keeping as many whole passages as the budget allows.

    ``ranked=False`` means the list is still in source order (the small-input
    shortcut before any relevance filter), so reordering it would be noise.
    """
    docs = docs if top_n is None else docs[:top_n]
    blocks = [prompt_family.pretty_print_docs([doc]) for doc in docs]

    limit = _token_budget(cfg)
    if limit <= 0:
        return "\n".join(_order_blocks(blocks, cfg) if ranked else blocks)

    fitted = fit_blocks(blocks, limit, getattr(cfg, "smart_llm_model", None))
    if fitted.truncated:
        logger.info(
            "Context token budget %s reached: kept %s passage(s), dropped %s.",
            limit, len(fitted.blocks), fitted.dropped,
        )
    kept = _order_blocks(fitted.blocks, cfg) if ranked else fitted.blocks
    return "\n".join(kept)


async def select_context(
    query: str,
    pages: list[dict],
    cfg,
    *,
    max_results: int = 10,
    prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
    cost_callback: Callable[[float], None] | None = None,
    embeddings: Callable[[], object] | None = None,
) -> str:
    """Context for ``query`` from ``pages`` (dicts with ``raw_content``, ``url``, ``title``).

    ``embeddings`` is a zero-argument factory, called only in embeddings mode,
    so nothing builds an embedding model unless that mode is chosen.
    """
    mode = resolve_context_filter(getattr(cfg, "context_filter", None))
    prompt_family = prompt_family or PromptFamily

    if mode == "none":
        return prompt_family.pretty_print_docs(_as_documents(pages))

    # Stage 1: drop duplicate pages before ranking, so copies do not take
    # slots from the passages they duplicate.
    if _dedup_enabled(cfg):
        pages = dedupe_pages(pages)

    # Little content needs no filtering, whatever the mode.
    total_chars = sum(len(str(p.get("raw_content") or "")) for p in pages)
    if total_chars < int(os.environ.get("COMPRESSION_THRESHOLD", "8000")) and len(pages) <= max_results:
        docs = _as_documents(pages)[:max_results]
        if _dedup_enabled(cfg):
            docs = dedupe_documents(docs, _dedup_threshold(cfg))
        return _render(docs, None, cfg, prompt_family, ranked=False)

    if mode == "summarize":
        try:
            from .summarize import HierarchicalSummarizer

            summary = await HierarchicalSummarizer(
                query=query,
                cfg=cfg,
                prompt_family=prompt_family,
                cost_callback=cost_callback,
            ).async_get_context(pages)
            # Derived context is still context: keep the summary inside the
            # same token budget the ranking modes respect.
            return fit_text(summary, _token_budget(cfg), getattr(cfg, "smart_llm_model", None))
        except Exception as e:
            logger.warning(f"Hierarchical summarization unavailable ({e}); using keyword ranking")

    # Stage 2: relevance filter. Every mode falls back to keyword ranking.
    docs, top_n = await _ranked_documents(
        mode, query, pages, cfg, max_results, prompt_family, cost_callback, embeddings,
    )

    # Stage 3: drop near-duplicate chunks the filter kept.
    if _dedup_enabled(cfg):
        docs = dedupe_documents(docs, _dedup_threshold(cfg))

    # Stage 4 (opt-in): compress the selected chunks into one summary.
    if _summarize_enabled(cfg):
        try:
            docs = await _summarize_documents(docs, query, cfg, prompt_family, cost_callback)
        except Exception as e:
            logger.warning(f"Context summarization unavailable ({e}); keeping selected passages")

    # Stage 5: keep the highest-ranked passages that fit the token budget.
    return _render(docs, top_n, cfg, prompt_family)


async def _ranked_documents(
    mode: str, query: str, pages: list[dict], cfg, max_results: int,
    prompt_family, cost_callback, embeddings,
) -> tuple[list[Document], int | None]:
    """Stage 2: the mode's ranking filter, with keyword ranking as the fallback."""
    if mode == "hybrid":
        # Fusion ranks both routes; a dead embedding route still leaves BM25
        # ranking (and the reverse never happens: BM25 needs nothing).
        docs = await hybrid_select(
            query,
            pages,
            cfg,
            max_results=max_results,
            prompt_family=prompt_family,
            cost_callback=cost_callback,
            embeddings=embeddings,
        )
        return docs, max_results

    if mode == "jev":
        try:
            docs = await JevContextCompressor(
                documents=pages, prompt_family=prompt_family
            ).async_select(query=query, max_results=max_results, cost_callback=cost_callback)
            return docs, max_results
        except JevError as e:
            logger.warning(f"Jev context filter unavailable ({e}); using keyword ranking")

    if mode == "embeddings":
        try:
            from .compression import ContextCompressor

            docs = await ContextCompressor(
                documents=pages,
                embeddings=embeddings() if embeddings else None,
                similarity_threshold=getattr(cfg, "similarity_threshold", None),
                prompt_family=prompt_family,
            ).async_select(query=query, max_results=max_results, cost_callback=cost_callback)
            return docs, max_results
        except Exception as e:
            logger.warning(f"Embedding context filter unavailable ({e}); using keyword ranking")

    docs = LexicalContextCompressor(
        documents=pages, relative_threshold=KEYWORD_RELATIVE_THRESHOLD, prompt_family=prompt_family,
    ).select(query, KEYWORD_MAX_RESULTS)
    return docs, KEYWORD_MAX_RESULTS


async def hybrid_select(
    query: str,
    pages: list[dict],
    cfg,
    *,
    max_results: int = 10,
    prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
    cost_callback: Callable[[float], None] | None = None,
    embeddings: Callable[[], object] | None = None,
) -> list[Document]:
    """BM25 and embedding candidates fused by reciprocal rank, best first.

    Both routes rank the full candidate pool (``HYBRID_CANDIDATES`` deep, not
    the final budget), so a chunk that is mediocre on one route but strong on
    the other still wins. ``HYBRID_LEXICAL_WEIGHT`` / ``HYBRID_SEMANTIC_WEIGHT``
    shift the balance, ``HYBRID_RRF_K`` changes how sharply top ranks dominate.

    The semantic route is best-effort: without an embedding provider, or when
    it fails, this returns BM25 ranking alone rather than no context.
    """
    depth = candidate_limit(max(int(max_results or 0), 1))
    lexical_docs = LexicalContextCompressor(
        documents=pages,
        relative_threshold=0.0,  # fusion decides what is relevant, not a cut-off
        prompt_family=prompt_family,
    ).select(query, depth)

    semantic_docs: list[Document] = []
    if embeddings is not None:
        try:
            semantic_docs = await async_semantic_ranks(
                query, pages, embeddings(), max_chunks=depth,
            )
        except Exception as e:  # noqa: BLE001 - one route may degrade
            logger.warning(f"Semantic route unavailable ({e}); fusing BM25 only")

    if semantic_docs:
        lexical_weight, semantic_weight = hybrid_weights()
        fused = reciprocal_rank_fusion(
            [lexical_docs, semantic_docs],
            k=rrf_k(),
            weights=[lexical_weight, semantic_weight],
        )
    else:
        # Reranking does not need the semantic route: with no embeddings the
        # BM25 order is still a candidate list worth a second look.
        fused = lexical_docs

    top_n = context_rerank_top_n(cfg)
    model = context_rerank_model(cfg)
    if model and top_n:
        fused = rerank_documents(query, fused, model=model, top_n=top_n)

    return fused[:max_results]


async def _summarize_documents(docs: list[Document], query: str, cfg, prompt_family, cost_callback) -> list[Document]:
    """Stage 4: compress the chunks a filter selected into one summary document."""
    from .summarize import HierarchicalSummarizer

    pages = [
        {
            "url": doc.metadata.get("source") or "",
            "title": doc.metadata.get("title") or "",
            "raw_content": doc.page_content,
        }
        for doc in docs
    ]
    if not pages:
        return docs

    text = await HierarchicalSummarizer(
        query=query, cfg=cfg, prompt_family=prompt_family, cost_callback=cost_callback,
    ).async_get_context(pages)
    if not text.strip():
        return docs
    return [Document(page_content=text, metadata={})]
