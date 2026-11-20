import re
from hashlib import sha256

from ..memory.draft import ClaimIssue


UNVERIFIED_CLAIM = "The available evidence is insufficient to verify this claim."
DEFAULT_MAX_CLAIM_SEARCHES = 1
MAX_SEARCHES_PER_SECTION = 3


def searchable_claims(state: dict) -> list[tuple[str, ClaimIssue]]:
    task = state.get("task") or {}
    per_claim = task.get("max_claim_searches", DEFAULT_MAX_CLAIM_SEARCHES)
    if not isinstance(per_claim, int):
        per_claim = DEFAULT_MAX_CLAIM_SEARCHES
    remaining = max(0, MAX_SEARCHES_PER_SECTION - state.get("claim_search_count", 0))
    return [
        (claim_id, issue)
        for claim_id, issue in (state.get("claim_issues") or {}).items()
        if issue["query"] and issue["attempts"] < per_claim
    ][:remaining]


def _quote_spans(body: str, quote: str) -> list[tuple[int, int]]:
    return [match.span() for match in re.finditer(re.escape(quote), body)]


def apply_claim_review(draft: dict, issues: dict[str, ClaimIssue], review: dict) -> None:
    """Apply verified candidates and keep unresolved quotes anchored in the draft."""
    reported = {}
    for item in review["issues"]:
        if not isinstance(item, dict):
            continue
        section, quote, query = (item.get(key) for key in ("section", "quote", "query"))
        if not isinstance(section, str) or not isinstance(quote, str) or not quote.strip():
            continue
        body = draft.get(section)
        if not isinstance(body, str) or quote not in body:
            continue
        claim_id = sha256(f"{section}\0{quote}".encode()).hexdigest()[:16]
        reported[claim_id] = (section, quote, query.strip() if isinstance(query, str) else "")

    for claim_id, (section, quote, query) in reported.items():
        if claim_id in issues:
            issues[claim_id] = {**issues[claim_id], "query": query or issues[claim_id]["query"]}
        else:
            issues[claim_id] = {
                "section": section, "quote": quote, "query": query,
                "attempts": 0, "evidence": "", "candidate": "",
            }

    for claim_id in review["accepted_claim_ids"]:
        issue = issues.get(claim_id) if isinstance(claim_id, str) else None
        if not issue or not issue["candidate"] or not issue["evidence"] or claim_id in reported:
            continue
        section = issue["section"]
        body = draft[section]
        other_spans = [
            span for other_id, other in issues.items()
            if other_id != claim_id and other["section"] == section
            for span in _quote_spans(body, other["quote"])
        ]
        if any(
            start < other_end and other_start < end
            for start, end in _quote_spans(body, issue["quote"])
            for other_start, other_end in other_spans
        ):
            continue
        draft[section] = body.replace(issue["quote"], issue["candidate"])
        del issues[claim_id]

    for claim_id, issue in issues.items():
        # A candidate that the reviewer did not approve must never reach Writer.
        issues[claim_id] = {**issue, "candidate": ""}


def finalize_claims(state: dict) -> tuple[dict, bool]:
    issues = state.get("claim_issues") or {}
    if not issues:
        return state["draft"], False
    draft = state["draft"].copy()
    for section, body in draft.items():
        if not isinstance(body, str):
            continue
        spans = sorted(
            span for issue in issues.values() if issue["section"] == section
            for span in _quote_spans(body, issue["quote"])
        )
        merged = []
        for start, end in spans:
            if merged and start < merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        for start, end in reversed(merged):
            body = body[:start] + UNVERIFIED_CLAIM + body[end:]
        draft[section] = body
    return draft, True
