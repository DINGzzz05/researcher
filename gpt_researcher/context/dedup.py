"""Removes duplicate and near-duplicate passages from context.

Search results repeat themselves: the same article is served under several
URLs, a syndicated copy differs only by a site header, and boilerplate from one
page turns up in another's extraction. Duplicate chunks cost tokens and bias
the writer toward whichever source was copied most often, so they are dropped
before ranking and again after chunk selection.

Two levels are handled:

* **pages** - identical URL, or identical normalized content under different
  URLs. One copy is kept; the rest are redundant for the writer.
* **chunks** - exact duplicates and near-duplicates (Jaccard similarity of
  word shingles at or above ``threshold``). Near-duplicate detection only runs
  on the small set of already-selected chunks, so it stays cheap.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, Sequence
from urllib.parse import urlsplit, urlunsplit

from langchain_core.documents import Document

_WHITESPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_SHINGLE_SIZE = 5
#: Default Jaccard similarity at which two chunks are considered the same text.
DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.9


def normalize_text(text: str | None) -> str:
    """Lowercase and collapse whitespace, so formatting doesn't defeat dedup."""
    return _WHITESPACE.sub(" ", (text or "").lower()).strip()


def fingerprint(text: str | None) -> str:
    """Stable hash of normalized ``text``."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def shingles(text: str, size: int = _SHINGLE_SIZE) -> frozenset[str]:
    """Word n-grams of ``text``, used to compare passages for near-duplication."""
    tokens = _NON_WORD.sub(" ", normalize_text(text)).split()
    if not tokens:
        return frozenset()
    if len(tokens) <= size:
        return frozenset({" ".join(tokens)})
    return frozenset(" ".join(tokens[i:i + size]) for i in range(len(tokens) - size + 1))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Jaccard similarity of two shingle sets, in [0, 1]."""
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    intersection = len(left & right)
    if not intersection:
        return 0.0
    return intersection / (len(left) + len(right) - intersection)


def normalize_url(url: str | None) -> str:
    """Canonical form for URL comparison: no fragment, no trailing slash, lowercase host."""
    url = (url or "").strip()
    if not url:
        return ""
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if not parts.scheme or not parts.netloc:
        return url
    return urlunsplit((
        parts.scheme.lower(),
        parts.netloc.lower(),
        parts.path.rstrip("/"),
        parts.query,
        "",
    ))


def _page_url(page: dict) -> str:
    return normalize_url(page.get("source") or page.get("url") or page.get("href") or "")


def dedupe_pages(pages: Sequence[dict] | None) -> list[dict]:
    """Drop pages whose URL or full text has already been seen, keeping order.

    The first copy wins: it is the one the retriever ranked highest. Pages with
    no content are deduplicated by URL only, since an empty hash would collapse
    every failed scrape into one.
    """
    kept: list[dict] = []
    seen_urls: set[str] = set()
    seen_content: set[str] = set()

    for page in pages or []:
        url = _page_url(page)
        content = page.get("raw_content") or ""
        body = fingerprint(content) if str(content).strip() else ""

        if url and url in seen_urls:
            continue
        if body and body in seen_content:
            continue

        if url:
            seen_urls.add(url)
        if body:
            seen_content.add(body)
        kept.append(page)

    return kept


def dedupe_documents(
    documents: Iterable[Document],
    threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
) -> list[Document]:
    """Drop exact and near-duplicate documents, keeping the first of each cluster.

    ``threshold`` is the minimum Jaccard similarity for two chunks to count as
    the same text; ``1.0`` (or higher) disables near-duplicate detection and
    leaves only exact duplicates removed.
    """
    threshold = float(threshold) if threshold is not None else 1.0
    near_duplicates = 0.0 < threshold < 1.0

    kept: list[Document] = []
    seen_exact: set[str] = set()
    signatures: list[frozenset[str]] = []
    sizes: list[int] = []

    for document in documents or []:
        text = document.page_content or ""
        key = fingerprint(text)
        if key in seen_exact:
            continue

        signature = shingles(text) if text.strip() else frozenset()
        if near_duplicates and signature:
            # Only compare with kept chunks whose size is close enough to reach
            # the threshold; a 100-word and a 5,000-word chunk cannot be 90% equal.
            duplicate = False
            for other, other_size in zip(signatures, sizes):
                if not other:
                    continue
                smaller, larger = sorted((len(signature), other_size))
                if larger and smaller / larger < threshold:
                    continue
                if jaccard(signature, other) >= threshold:
                    duplicate = True
                    break
            if duplicate:
                continue

        seen_exact.add(key)
        kept.append(document)
        signatures.append(signature)
        sizes.append(len(signature))

    return kept
