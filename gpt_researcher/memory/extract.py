"""Turning a finished report into what a later run actually needs.

A memory record carries two pieces that must be *derived* from the run, not
stored raw:

* the **summary**. It used to be ``report[:800]`` - the title and the opening
  paragraph - which told the next run what was asked rather than what was
  found. It is now selected where the writer assembles the report: the
  conclusion first (that is where the findings are), then the opening
  paragraph (the scope), then the body paragraphs that match the query, cut at
  a sentence boundary so the result reads as prose rather than as a fragment.
* the **sources**. ``visited_urls`` is a raw set: redirect wrappers, tracking
  parameters, fragments, the same page under several query strings and the odd
  ``mailto:``. Stored as-is they crowd the per-record budget with duplicates
  and make recall look sloppier than the run was. ``clean_sources`` unwraps,
  canonicalises, drops tracking noise, dedupes, then spreads the survivors
  across domains instead of taking twelve pages from one site.

Both are dependency-free and cheap: this runs once per finished report.
"""

from __future__ import annotations

import base64
import binascii
import re
from typing import Iterable, Sequence
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

from ..context.dedup import jaccard
from ..context.lexical import bm25_scores, tokenize

#: Characters kept from the selected parts of a report.
DEFAULT_SUMMARY_CHARS = 800
#: Source URLs kept per memory record.
DEFAULT_MAX_SOURCES = 12
#: Sources kept per host, so one site cannot fill the whole budget.
DEFAULT_MAX_PER_DOMAIN = 3

_SUMMARY_HEADING = re.compile(
    r"(conclusion|summary|key\s+findings|takeaways?|tl;?dr|结论|总结|小结|要点|摘要)",
    re.IGNORECASE,
)
_INTRO_HEADING = re.compile(
    r"(introduction|overview|background|引言|概述|背景)",
    re.IGNORECASE,
)
_REFERENCES_HEADING = re.compile(
    r"(references|sources|bibliography|citations|参考文献|引用来源)",
    re.IGNORECASE,
)

_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]*", re.MULTILINE)
_MD_RULE = re.compile(r"^[ \t]*([-*_][ \t]*){3,}$", re.MULTILINE)
_MD_TABLE_RULE = re.compile(r"^[ \t]*\|?[\s:|-]+\|?[ \t]*$", re.MULTILINE)
_MD_LIST = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+", re.MULTILINE)
_MD_EMPHASIS = re.compile(r"(\*\*|__|\*|_|`)")
_HEADING_LINE = re.compile(r"^[ \t]{0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_WHITESPACE = re.compile(r"\s+")
_SENTENCE_BREAK = re.compile(r"(?<=[。！？；])|(?<=[.!?])\s+")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")

#: Query parameters that only track the visit, never identify the document.
_TRACKING_PARAMS = frozenset(
    """
    utm_source utm_medium utm_campaign utm_term utm_content utm_id utm_name
    utm_cid utm_reader gclid gclsrc dclid fbclid msclkid yclid twclid igshid
    mc_cid mc_eid _hsenc _hsmi mkt_tok ref ref_src ref_url spm scm from smid
    session_id
    """.split()
)
_TRACKING_PREFIXES = ("utm_", "pk_", "mtm_", "hsa_", "_ga")

#: Hosts whose result pages are not sources in their own right.
_SEARCH_HOSTS = frozenset(
    """
    google.com bing.com duckduckgo.com baidu.com so.com sogou.com yandex.com
    search.yahoo.com search.brave.com
    """.split()
)
_SEARCH_PATHS = frozenset({"", "/", "/search", "/s", "/web", "/html", "/webhp"})


# --------------------------------------------------------------------------- summary


def _plain(markdown: str) -> str:
    """Markdown to readable prose: links keep their label, syntax disappears."""
    text = _MD_IMAGE.sub(" ", markdown or "")
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_RULE.sub(" ", text)
    text = _MD_TABLE_RULE.sub(" ", text)
    text = _MD_HEADING.sub(" ", text)
    text = _MD_LIST.sub(" ", text)
    text = _MD_EMPHASIS.sub("", text)
    text = text.replace("|", " ")
    return _WHITESPACE.sub(" ", text).strip()


def _norm(text: str) -> str:
    return _WHITESPACE.sub(" ", (text or "").lower()).strip()


def _sections(markdown: str) -> list[tuple[int, str, str]]:
    """``(level, heading, body)`` for every heading block, in document order."""
    rows: list[tuple[int, str, str]] = []
    level, heading, body = 0, "", []
    for line in (markdown or "").splitlines():
        match = _HEADING_LINE.match(line)
        if match:
            rows.append((level, heading, "\n".join(body)))
            level, heading, body = len(match.group(1)), match.group(2).strip(), []
        else:
            body.append(line)
    rows.append((level, heading, "\n".join(body)))
    return [row for row in rows if row[1] or row[2].strip()]


def _paragraphs(text: str) -> list[str]:
    return [p for p in (_plain(chunk) for chunk in _PARAGRAPH_BREAK.split(text or "")) if len(p) > 2]


def _first_sentences(text: str, count: int = 2) -> str:
    sentences = [s.strip() for s in _SENTENCE_BREAK.split(text or "") if s.strip()]
    return " ".join(sentences[:count]).strip()

def _truncate(text: str, limit: int) -> str:
    """Cut at the last sentence end inside ``limit``, else at a word boundary."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text

    window = text[:limit]
    cut = max(window.rfind(mark) for mark in ".!?。！？；;")
    if cut >= int(limit * 0.5):
        return window[: cut + 1].strip()

    space = window.rfind(" ")
    if space > 0:
        window = window[:space]
    return window.rstrip(" ,;，；:：") + "…"


def _join_limited(segments: Iterable[str], limit: int) -> str:
    """Join segments in priority order, stopping when the budget runs out."""
    picked: list[str] = []
    seen: set[str] = set()
    used = 0

    for segment in segments:
        segment = (segment or "").strip()
        if not segment:
            continue
        key = _norm(segment)
        if not key or key in seen:
            continue
        seen.add(key)

        separator = 1 if picked else 0
        room = limit - used - separator
        if room < 40:
            break
        if len(segment) > room:
            segment = _truncate(segment, room)
            if not segment:
                break
        picked.append(segment)
        used += len(segment) + separator

    return " ".join(picked)


def _lead_paragraph(sections: Sequence[tuple[int, str, str]]) -> str:
    """First real paragraph of the report: title headings have no body."""
    for _, _, body in sections:
        paragraphs = _paragraphs(body)
        if paragraphs:
            return paragraphs[0]
    return ""


def _summary_from_markdown(query: str, report: str, limit: int) -> str:
    sections = _sections(report)
    if not sections:
        return _truncate(_plain(report), limit)

    summary_sections = [s for s in sections if s[1] and _SUMMARY_HEADING.search(s[1])]
    intro_sections = [s for s in sections if s[1] and _INTRO_HEADING.search(s[1])]

    segments: list[str] = []
    if summary_sections:
        segments.extend(_plain(body) for _, _, body in summary_sections)
    else:
        # Without a conclusion the opening paragraph is the best stand-in: it
        # states the scope, which the stored query alone does not.
        lead = _lead_paragraph(sections)
        if lead:
            segments.append(lead)

    for _, _, body in intro_sections:
        intro = _first_sentences(_plain(body))
        if intro:
            segments.append(intro)

    body_paragraphs: list[str] = []
    for _, heading, body in sections:
        if heading and (_SUMMARY_HEADING.search(heading) or _INTRO_HEADING.search(heading)
                        or _REFERENCES_HEADING.search(heading)):
            continue
        body_paragraphs.extend(_paragraphs(body))

    if body_paragraphs:
        scores = bm25_scores(query, body_paragraphs)
        order = sorted(range(len(body_paragraphs)), key=lambda i: (-scores[i], i))
        segments.extend(body_paragraphs[i] for i in order[:3])

    return _join_limited(segments, limit) or _truncate(_plain(report), limit)


def _summary_from_parts(query: str, parts: Sequence[tuple[str, str]], limit: int) -> str:
    """Summary from the report pieces handed over by the writer at assembly time."""
    summary, intro, body = [], [], []
    for label, text in parts:
        key = (label or "").lower()
        if _SUMMARY_HEADING.search(key):
            plain = _plain(text)
            if plain:
                summary.append(plain)
        elif _INTRO_HEADING.search(key):
            plain = _first_sentences(_plain(text))
            if plain:
                intro.append(plain)
        else:
            body.extend(_paragraphs(text))

    segments = summary + intro
    if body:
        scores = bm25_scores(query, body)
        order = sorted(range(len(body)), key=lambda i: (-scores[i], i))
        segments.extend(body[i] for i in order[:3])

    return _join_limited(segments, limit)


def select_summary(
    query: str,
    report: str = "",
    *,
    parts: Sequence[tuple[str, str]] | None = None,
    limit: int = DEFAULT_SUMMARY_CHARS,
) -> str:
    """The memory summary for a run: findings first, scope second.

    ``parts`` are the ``(label, text)`` pieces the writer joined into the report
    (``introduction`` / ``body`` / ``conclusion``). When they are available the
    summary is taken from them directly; otherwise it is parsed out of the
    final markdown, which is what a single-blob writer produces.
    """
    limit = max(80, int(limit or DEFAULT_SUMMARY_CHARS))
    if parts:
        summary = _summary_from_parts(query, parts, limit)
        if summary:
            return summary
    return _summary_from_markdown(query, report or "", limit)


# ----------------------------------------------------------------------------- facts

#: Claims kept from one report. Enough to carry the findings, few enough that
#: the fact store stays a summary of the research rather than a copy of it.
DEFAULT_FACT_LIMIT = 6

_FACT_SIGNAL = re.compile(
    r"(\d|%|percent|increase|decrease|grew|grow|decline|higher|lower|more than|"
    r"less than|because|due to|driven by|led to|results in|compared|whereas|"
    r"raises|improves|reduces|throughput|latency|boost|scales|limits|constrains|"
    r"constraint|bottleneck|"
    r"达到|增长|下降|高于|低于|超过|占比|因为|由于|导致|因此|相比|提升|降低|成本|"
    r"revenue|market share|adoption|policy|regulation)",
    re.IGNORECASE,
)
_FACT_META = re.compile(
    r"\b(this report|this section|in this (report|section|article)|we will|"
    r"the following|below we|as mentioned)\b|(本文|本报告|本节|接下来|如上所述)",
    re.IGNORECASE,
)
_LEAD_TRANSITION = re.compile(
    r"^(however|moreover|furthermore|therefore|thus|overall|additionally|"
    r"in conclusion|in summary|besides|此外|因此|总体来看|综上|总的来说)[,，:：]?\s*",
    re.IGNORECASE,
)
_LATIN_ENTITY = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9]*(?:[.+#-][A-Za-z0-9]+)*|[a-z]+[A-Z][A-Za-z0-9]*)\b"
)
_QUOTED = re.compile(r"[《“\"]([^》”\"]{2,40})[》”\"]")
_ENTITY_STOP = frozenset(
    """
    the this that these those however moreover furthermore therefore thus overall
    report introduction conclusion summary section table figure source sources
    according research study data results
    """.split()
)


def split_sentences(text: str) -> list[str]:
    """Sentences in both scripts: CJK punctuation splits anywhere, Latin needs a space."""
    return [s.strip() for s in _SENTENCE_BREAK.split(text or "") if s.strip()]


def extract_entities(text: str, *, extra: Sequence[str] = (), limit: int = 6) -> list[str]:
    """Named anchors for entity-based recall: proper nouns, acronyms, quoted terms.

    Unlike an LLM extractor this cannot tell "Python" the language from
    "python" the snake, so the anchor is only used to *boost* a hit, never to
    filter one out. Chinese text has no capitalisation to key on, so the
    caller passes the query's own terms in ``extra`` and those act as topical
    anchors.
    """
    found: list[str] = []

    # Quoted names first: they are explicitly marked as names, and the cap
    # below should not let loose capitals crowd them out.
    found.extend(quoted.strip() for quoted in _QUOTED.findall(text or ""))
    for match in _LATIN_ENTITY.finditer(text or ""):
        name = match.group(0).strip(".,;:()[]")
        if len(name) >= 2 and any(ch.isalpha() for ch in name):
            found.append(name)
    found.extend(str(term).strip() for term in extra)

    entities: list[str] = []
    seen: set[str] = set()
    for name in found:
        key = name.lower()
        if not name or len(name) > 40 or key in seen or key in _ENTITY_STOP:
            continue
        seen.add(key)
        entities.append(name)
        if len(entities) >= limit:
            break
    return entities


def _clean_claim(sentence: str) -> str:
    raw = (sentence or "").strip()
    claim = _LEAD_TRANSITION.sub("", raw)
    if claim != raw and claim:
        # "However, cobalt..." -> "Cobalt...", so the stored claim reads as a
        # statement rather than a fragment of one.
        claim = claim[0].upper() + claim[1:]
    claim = _WHITESPACE.sub(" ", claim).strip(" -–—*")
    return _truncate(claim, 320)


def _is_fact(claim: str) -> bool:
    if len(claim) < 12 or claim.endswith("?"):
        return False
    if _FACT_META.search(claim):
        return False
    if len(set(tokenize(claim))) < 3 and len(claim) < 40:
        return False
    return bool(_FACT_SIGNAL.search(claim))


def _facts_from_text(text: str, limit: int, query_terms: set[str] | None = None) -> list[str]:
    query_terms = query_terms or set()
    candidates: list[tuple[int, str]] = []
    seen: list[frozenset[str]] = []

    for sentence in split_sentences(_plain(text)):
        claim = _clean_claim(sentence)
        if not _is_fact(claim):
            continue
        tokens = frozenset(tokenize(claim))
        if any(jaccard(tokens, previous) >= 0.8 for previous in seen):
            continue
        seen.append(tokens)
        candidates.append((len(query_terms.intersection(tokens)), claim))

    ranked = sorted(enumerate(candidates), key=lambda item: (-item[1][0], item[0]))
    return [claim for _, (_, claim) in ranked[:limit]]


def extract_facts(report: str, query: str = "", *, limit: int = DEFAULT_FACT_LIMIT) -> list[str]:
    """Atomic claims worth remembering from a finished report.

    Facts come from where findings live - the conclusion/summary sections, or
    the lead paragraph when the report has none - so the semantic layer does
    not inherit the report's middle-of-document filler. Selection is lexical:
    a sentence qualifies when it carries a signal (a number, a comparison, a
    cause, a policy/regulation) and is not report meta-talk.
    """
    if limit <= 0:
        return []

    sections = _sections(report or "")
    if not sections:
        return _facts_from_text(report or "", limit, set(tokenize(query)))

    preferred = [
        section for section in sections
        if section[1] and _SUMMARY_HEADING.search(section[1])
    ]
    if not preferred:
        # No conclusion: the introduction states scope, and if there is not
        # even that, the opening sections carry the report's main points.
        preferred = [
            section for section in sections
            if section[1] and _INTRO_HEADING.search(section[1])
        ]
    if not preferred:
        preferred = [section for section in sections if section[2].strip()][:2]

    text = " ".join(_plain(body) for _, _, body in preferred)
    return _facts_from_text(text or _plain(report or ""), limit, set(tokenize(query)))


# --------------------------------------------------------------------------- sources


def _unwrap_once(url: str) -> str:
    """Follow one search-engine / social redirect wrapper, if ``url`` is one."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return ""
    host = (parts.hostname or "").lower()
    query = dict(parse_qsl(parts.query, keep_blank_values=False))

    if _host_matches(host, "google.com") and parts.path.rstrip("/") == "/url":
        return next((query[k] for k in ("q", "url", "imgurl") if query.get(k)), "")
    if _host_matches(host, "bing.com") and parts.path.rstrip("/") == "/ck/a":
        target = query.get("u", "")
        return _unwrap_bing(target) or target
    if _host_matches(host, "duckduckgo.com") and parts.path.startswith("/l"):
        return query.get("uddg", "")
    if _host_matches(host, "l.facebook.com") and parts.path.startswith("/l.php"):
        return query.get("u", "")
    return ""


def _unwrap_bing(value: str) -> str:
    """Bing encodes the target as ``a1`` + urlsafe base64."""
    if not value.startswith("a1"):
        return ""
    payload = value[2:]
    try:
        raw = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
    except (binascii.Error, ValueError):
        return ""
    return raw.decode("utf-8", "ignore")


def _host_matches(host: str, base: str) -> bool:
    return host == base or host.endswith("." + base)


def _is_tracking(key: str) -> bool:
    key = key.lower()
    return key in _TRACKING_PARAMS or key.startswith(_TRACKING_PREFIXES)


def _canonical(url: str) -> str:
    """One clean spelling of a source URL, or ``""`` when it is not a source."""
    url = (url or "").strip()
    if not url:
        return ""
    if url.startswith("//"):
        url = "https:" + url
    if "://" not in url:
        if url.startswith(("mailto:", "javascript:", "tel:", "data:", "#")):
            return ""
        if "." not in url.split("/", 1)[0]:
            return ""
        url = "https://" + url

    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return ""

    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        return ""

    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        return ""
    if host.startswith("www."):
        host = host[4:]

    if _is_search_page(host, parts.path):
        return ""

    netloc = host
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"

    path = re.sub(r"/{2,}", "/", parts.path or "")
    if path == "/":
        path = ""
    else:
        path = path.rstrip("/")

    params = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=False) if not _is_tracking(k)]
    params.sort()
    query = urlencode(params, doseq=False)

    return urlunsplit((scheme, netloc, path, query, ""))


def _is_search_page(host: str, path: str) -> bool:
    if not any(_host_matches(host, base) for base in _SEARCH_HOSTS):
        return False
    return path.rstrip("/") in _SEARCH_PATHS or path.rstrip("/").startswith(("/search", "/s?"))


def _dedupe_key(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.netloc}{parts.path.lower()}?{parts.query}"


def clean_sources(
    urls: Iterable[str],
    *,
    query: str = "",
    limit: int = DEFAULT_MAX_SOURCES,
    per_domain: int = DEFAULT_MAX_PER_DOMAIN,
) -> list[str]:
    """Canonical, deduped, query-relevant source URLs for one memory record.

    Order is relevance to ``query`` (keyword overlap on the URL), then original
    order, with at most ``per_domain`` URLs per host so a single site cannot
    take over the record.
    """
    limit = max(0, int(limit or 0))
    if not limit:
        return []
    per_domain = max(1, int(per_domain or 1))

    terms = set(tokenize(query or ""))
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()

    for raw in urls or []:
        url = _unwrap_url(raw) if isinstance(raw, str) else ""
        if not url:
            continue
        key = _dedupe_key(url)
        if key in seen:
            continue
        seen.add(key)
        haystack = tokenize(unquote(url))
        score = len(terms.intersection(haystack)) if terms else 0
        candidates.append((url, score))

    ranked = sorted(enumerate(candidates), key=lambda item: (-item[1][1], item[0]))

    picked: list[str] = []
    per_host: dict[str, int] = {}
    for _, (url, _score) in ranked:
        if len(picked) >= limit:
            break
        host = urlsplit(url).hostname or ""
        if per_host.get(host, 0) >= per_domain:
            continue
        per_host[host] = per_host.get(host, 0) + 1
        picked.append(url)
    return picked


def _unwrap_url(raw: str) -> str:
    """Unwrap redirect chains, then canonicalise; ``""`` when it is not a source."""
    url = (raw or "").strip()
    for _ in range(3):
        target = _unwrap_once(url)
        if not target:
            break
        url = target.strip()
    return _canonical(url)
