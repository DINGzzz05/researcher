from typing import List, NotRequired, TypedDict


class ClaimIssue(TypedDict):
    section: str
    quote: str
    query: str
    attempts: int
    evidence: str
    candidate: str


class DraftState(TypedDict):
    task: dict
    topic: str
    sibling_sections: List[str]
    draft: dict
    search_errors: NotRequired[list[dict]]
    review: str
    revision_notes: str
    draft_revision_count: int
    claim_issues: NotRequired[dict[str, ClaimIssue]]
    claim_search_count: NotRequired[int]
