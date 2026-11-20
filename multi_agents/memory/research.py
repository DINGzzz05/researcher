from typing import TypedDict, List, Annotated, NotRequired
import operator


class ResearchState(TypedDict):
    task: dict
    initial_research: str
    sections: List[str]
    research_data: List[dict]
    search_errors: NotRequired[list[dict]]
    unresolved_claim_sections: NotRequired[List[int]]
    # Full section drafts live in section_drafts_dir; section_drafts maps a
    # section title to its file name, and research_data carries only summaries.
    section_drafts_dir: NotRequired[str]
    section_drafts: NotRequired[dict[str, str]]
    human_feedback: str
    plan_revision_count: int
    # Report layout
    title: str
    headers: dict
    date: str
    table_of_contents: str
    introduction: str
    conclusion: str
    sources: List[str]
    report: str
    fact_check_notes: str
    diagrams: List[str]
    fact_check_revision_count: int
