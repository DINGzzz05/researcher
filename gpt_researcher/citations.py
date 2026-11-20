"""Claim-level citation tracing for generated reports.

GPT Researcher already asks the writer to put a markdown link at the end of
every sourced sentence, and ``add_references`` lists the URLs a run visited.
What was missing was a check: did the report actually cite the sources it was
given, and does every cited URL come from those sources? This module answers
both questions without an LLM.

``verify_citations`` walks the report, pairs each substantive sentence with the
URLs cited in it, and reports:

* the claim -> sources mapping (the traceability structure),
* sentences that make a claim with no citation,
* URLs cited by the report that were never retrieved (hallucinated sources),
* retrieved URLs that never appear in the report.

Parsing is deliberately conservative: headings, code blocks and the References
section are excluded, and sentences shorter than ``min_claim_chars`` are not
counted as claims, so the audit does not flag labels or table separators.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable
from urllib.parse import urlsplit, urlunsplit

#: Inline markdown link, but not an image (``![alt](src)``). One level of
#: parentheses inside the URL is allowed, so Wikipedia-style URLs survive.
_LINK = re.compile(
    r"(?<!!)\[([^\]]*)\]\(((?:[^()\s]|\([^()\s]*\))+)(?:\s+\"[^\"]*\")?\)"
)
_FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`]*`")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s")
_REFERENCES_HEADING = re.compile(
    r"^#{1,6}\s*(?:references|bibliography|sources|works\s+cited)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_TABLE_SEPARATOR = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
_MARKDOWN_NOISE = re.compile(r"[*_`>#|]+")
_LETTER = re.compile(r"[A-Za-z\u00c0-\u024f\u4e00-\u9fff]")


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


def _plain_text(markdown: str) -> str:
    """``markdown`` with link targets and formatting removed, for length checks."""
    text = _LINK.sub(lambda match: match.group(1), markdown)
    text = _MARKDOWN_NOISE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class Claim:
    """One substantive sentence and the URLs it cites."""

    text: str
    sources: tuple[str, ...] = ()


@dataclass
class CitationAudit:
    """Result of auditing a report's citations against its retrieved sources."""

    claims: list[Claim] = field(default_factory=list)
    uncited_claims: list[str] = field(default_factory=list)
    cited_sources: list[str] = field(default_factory=list)
    unknown_sources: list[str] = field(default_factory=list)
    unused_sources: list[str] = field(default_factory=list)

    @property
    def total_claims(self) -> int:
        return len(self.claims)

    @property
    def cited_claims(self) -> int:
        return sum(1 for claim in self.claims if claim.sources)

    @property
    def coverage(self) -> float:
        """Share of claims that carry at least one citation, in [0, 1]."""
        if not self.claims:
            return 1.0
        return self.cited_claims / len(self.claims)

    def to_dict(self) -> dict:
        """JSON-serializable view, for API responses and logging."""
        return {
            "total_claims": self.total_claims,
            "cited_claims": self.cited_claims,
            "coverage": round(self.coverage, 3),
            "citations": [
                {"claim": claim.text, "sources": list(claim.sources)}
                for claim in self.claims
            ],
            "uncited_claims": list(self.uncited_claims),
            "cited_sources": list(self.cited_sources),
            "unknown_sources": list(self.unknown_sources),
            "unused_sources": list(self.unused_sources),
        }


def _cut_references(markdown: str) -> str:
    match = _REFERENCES_HEADING.search(markdown)
    if match:
        return markdown[: match.start()]
    return markdown


def _split_blocks(markdown: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n", markdown) if block.strip()]


def _split_sentences(block: str) -> list[str]:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines:
        return []

    # A list item or table row is one claim per line; a wrapped paragraph is
    # joined back together so its citation on the last visual line still
    # belongs to the sentence it ends.
    listy = any(
        re.match(r"^\s*(?:[-*+]|\d+[.)])\s+", line) or line.startswith("|")
        for line in lines
    )
    units = lines if listy else [" ".join(lines)]

    sentences: list[str] = []
    for unit in units:
        sentences.extend(part.strip() for part in _SENTENCE_BREAK.split(unit) if part.strip())
    return sentences


def extract_claims(report: str | None, min_claim_chars: int = 60) -> list[Claim]:
    """Substantive sentences in ``report``, each with the URLs cited in it.

    ``min_claim_chars`` filters out headings, short labels and link-only lines
    that are not claims. URLs are only the http(s) links written inline.
    """
    if not report:
        return []

    text = _FENCED_CODE.sub(" ", str(report))
    text = _HTML_COMMENT.sub(" ", text)
    text = _INLINE_CODE.sub(" ", text)
    text = _cut_references(text)

    claims: list[Claim] = []
    for block in _split_blocks(text):
        if _HEADING.match(block) or block.startswith("!["):
            continue
        if _TABLE_SEPARATOR.match(block):
            continue
        for sentence in _split_sentences(block):
            plain = _plain_text(sentence)
            if not _LETTER.search(plain):
                continue
            urls = tuple(dict.fromkeys(
                url for url in (match[1] for match in _LINK.findall(sentence))
                if url.lower().startswith(("http://", "https://"))
            ))
            # A sentence that carries a citation is a claim by construction.
            # Length only decides whether an uncited sentence is substantive.
            if not urls and len(plain) < min_claim_chars:
                continue
            claims.append(Claim(text=sentence, sources=urls))
    return claims


def verify_citations(
    report: str | None,
    allowed_urls: Iterable[str] | None = None,
    *,
    min_claim_chars: int = 60,
) -> CitationAudit:
    """Audit ``report`` against ``allowed_urls`` (the run's retrieved sources).

    A URL is known when it matches an allowed URL after normalization (fragment
    and trailing slash ignored). Unknown URLs are usually invented or
    mis-copied links, which is exactly what citation tracing should catch.
    """
    claims = extract_claims(report, min_claim_chars=min_claim_chars)

    allowed_by_key: dict[str, str] = {}
    for url in allowed_urls or []:
        if not url:
            continue
        allowed_by_key.setdefault(normalize_url(url), url)

    cited_keys: set[str] = set()
    unknown: list[str] = []
    for claim in claims:
        for url in claim.sources:
            key = normalize_url(url)
            if key in allowed_by_key:
                cited_keys.add(key)
            elif url not in unknown:
                unknown.append(url)

    return CitationAudit(
        claims=claims,
        uncited_claims=[claim.text for claim in claims if not claim.sources],
        cited_sources=sorted({allowed_by_key[key] for key in cited_keys}),
        unknown_sources=unknown,
        unused_sources=sorted(
            {url for key, url in allowed_by_key.items() if key not in cited_keys}
        ),
    )


def cited_urls(report: str | None, allowed_urls: Iterable[str] | None = None) -> set[str]:
    """The allowed URLs that ``report`` actually cites (original spelling)."""
    return set(verify_citations(report, allowed_urls).cited_sources)
