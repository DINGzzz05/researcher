import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest

from multi_agents.agents import researcher, reviewer, reviser
from multi_agents.agents.claim_review import UNVERIFIED_CLAIM
from multi_agents.agents.editor import EditorAgent
from multi_agents.agents.publisher import PublisherAgent
from multi_agents.agents.researcher import ResearchAgent
from multi_agents.agents.writer import WriterAgent


CLAIM = "Elasticsearch is three times faster than PostgreSQL."
OTHER_TEXT = "Both engines support full-text search."
QUERY = "Elasticsearch PostgreSQL comparable full-text benchmark"
SOURCE = "https://example.com/benchmark"
REPLACEMENT = f"The benchmark reported a 10% difference on its workload ([study]({SOURCE}))."


@pytest.fixture
def chapter(monkeypatch):
    monkeypatch.setattr(ResearchAgent, "run_depth_research", AsyncMock(return_value={
        "draft": {"Search": f"{OTHER_TEXT}\n\n{CLAIM}"},
    }))
    state = {"task": {"follow_guidelines": False}, "sections": ["Search"]}
    return state


def audit(*, issues=(), accepted=()):
    return {"issues": list(issues), "accepted_claim_ids": list(accepted), "notes": None}


def claim_issue():
    return {"section": "Search", "quote": CLAIM, "query": QUERY}


def test_claim_research_reuses_configured_source_without_writing_report(monkeypatch):
    evidence = f"Source: {SOURCE}\nThe measured difference was 10%."
    client = Mock(conduct_research=AsyncMock(return_value=[evidence]), write_report=AsyncMock())
    factory = Mock(return_value=client)
    monkeypatch.setattr(researcher, "GPTResearcher", factory)

    assert asyncio.run(ResearchAgent().research_claim(QUERY, {"source": "local"})) == evidence
    assert factory.call_args.kwargs["query"] == QUERY
    assert factory.call_args.kwargs["report_source"] == "local"
    client.write_report.assert_not_awaited()


def test_reviser_only_searches_the_flagged_claim_and_commits_approved_candidate(monkeypatch, chapter):
    async def review_model(prompt, **kwargs):
        issues = json.loads(prompt[1]["content"])["open_claims"]
        if not issues:
            return audit(issues=[claim_issue()])
        claim_id, issue = next(iter(issues.items()))
        assert issue["candidate"] == REPLACEMENT
        assert SOURCE in issue["evidence"]
        return audit(accepted=[claim_id])

    async def revise_model(prompt, **kwargs):
        claim_id, item = next(iter(json.loads(prompt[1]["content"]).items()))
        assert item["quote"] == CLAIM
        return {claim_id: REPLACEMENT}

    search = AsyncMock(return_value=f"Source: {SOURCE}\nThe measured difference was 10%.")
    monkeypatch.setattr(ResearchAgent, "research_claim", search)
    monkeypatch.setattr(reviewer, "call_model", review_model)
    monkeypatch.setattr(reviser, "call_model", revise_model)

    result = asyncio.run(EditorAgent().run_parallel_research(chapter))

    search.assert_awaited_once_with(QUERY, chapter["task"])
    assert result["research_data"] == [{"Search": f"{OTHER_TEXT}\n\n{REPLACEMENT}"}]
    assert result["unresolved_claim_sections"] == []


@pytest.mark.parametrize("fail_search", [False, True])
def test_unresolved_claim_is_removed_before_writer_and_marked_by_publisher(monkeypatch, chapter, fail_search):
    # Omitting an existing claim from the next review is not an explicit approval.
    monkeypatch.setattr(reviewer, "call_model", AsyncMock(side_effect=[audit(issues=[claim_issue()]), audit()]))
    if fail_search:
        search = AsyncMock(side_effect=TimeoutError)
    else:
        search = AsyncMock(return_value=f"Source: {SOURCE}\nUnrelated excerpt.")
    monkeypatch.setattr(ResearchAgent, "research_claim", search)

    async def revise_model(prompt, **kwargs):
        claim_id = next(iter(json.loads(prompt[1]["content"])))
        return {claim_id: REPLACEMENT}

    monkeypatch.setattr(reviser, "call_model", revise_model)
    result = asyncio.run(EditorAgent().run_parallel_research(chapter))
    search.assert_awaited_once_with(QUERY, chapter["task"])

    assert result["research_data"] == [{"Search": f"{OTHER_TEXT}\n\n{UNVERIFIED_CLAIM}"}]
    assert result["unresolved_claim_sections"] == [0]
    with pytest.MonkeyPatch.context() as patch:
        writer_model = AsyncMock(return_value={})
        patch.setattr("multi_agents.agents.writer.call_model", writer_model)
        asyncio.run(WriterAgent().write_sections({**chapter, **result}))
        writer_prompt = writer_model.await_args.args[0][1]["content"]
        assert CLAIM not in writer_prompt and REPLACEMENT not in writer_prompt
    report = PublisherAgent(output_dir=".").generate_layout(result)
    assert "[证据不足 / Insufficient evidence]" in report
    assert OTHER_TEXT in report and CLAIM not in report and REPLACEMENT not in report


def test_revision_cap_still_removes_flagged_claim(monkeypatch, chapter):
    chapter["task"]["max_draft_revisions"] = 0
    search = AsyncMock()
    monkeypatch.setattr(ResearchAgent, "research_claim", search)
    monkeypatch.setattr(reviewer, "call_model", AsyncMock(return_value=audit(issues=[claim_issue()])))

    result = asyncio.run(EditorAgent().run_parallel_research(chapter))

    search.assert_not_awaited()
    assert result["research_data"] == [{"Search": f"{OTHER_TEXT}\n\n{UNVERIFIED_CLAIM}"}]
    assert result["unresolved_claim_sections"] == [0]
