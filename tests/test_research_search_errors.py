from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import requests
from langchain_core.tools import StructuredTool, ToolException

from gpt_researcher import GPTResearcher
from gpt_researcher.actions.query_processing import invoke_retriever_search
from gpt_researcher.llm_provider.generic.base import GenericLLMProvider
from gpt_researcher.mcp.research import MCPResearchSkill
from gpt_researcher.retrievers.errors import InsufficientSearchEvidence, SearchError
from gpt_researcher.retrievers.tavily.tavily_search import TavilySearch
from gpt_researcher.skills.researcher import ResearchConductor


def http_error(status, retry_after=None):
    response = requests.Response()
    response.status_code = status
    if retry_after is not None:
        response.headers["Retry-After"] = retry_after
    return requests.HTTPError("internal response details", response=response)


def test_tavily_distinguishes_empty_results_from_a_malformed_response(monkeypatch):
    retriever = TavilySearch("query", headers={"tavily_api_key": "test-key"})
    monkeypatch.setattr(retriever, "_search", Mock(side_effect=[{"results": []}, {"results": {}}]))
    assert retriever.search_for_research() == []
    with pytest.raises(SearchError) as caught:
        retriever.search_for_research()
    assert caught.value.code == "invalid_response"


async def test_rate_limit_honors_retry_after_and_stops_at_the_attempt_limit(monkeypatch):
    retriever = TavilySearch("query", headers={"tavily_api_key": "test-key"})
    request = Mock(side_effect=http_error(429, "4"))
    delay = AsyncMock()
    record = AsyncMock()
    monkeypatch.setattr(retriever, "_search", request)
    monkeypatch.setattr("gpt_researcher.retrievers.errors.asyncio.sleep", delay)

    assert await invoke_retriever_search(retriever, 5, on_error=record) == []
    assert request.call_count == 3
    assert [call.args[0] for call in delay.await_args_list] == [4, 4]
    error = record.await_args.args[0]
    assert (error.provider, error.code, error.attempts) == ("Tavily", "rate_limited", 3)
    assert isinstance(error.__cause__, requests.HTTPError)


async def test_initial_and_subquery_failures_preserve_other_sources(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    request = Mock(side_effect=http_error(401))
    monkeypatch.setattr(TavilySearch, "_search", request)
    monkeypatch.setattr("gpt_researcher.skills.researcher.plan_research_outline", AsyncMock(return_value=["detail"]))

    class KnowledgeBase:
        requires_scraping = False

        def __init__(self, query, query_domains=None):
            pass

        def search(self, max_results):
            return [{"href": "https://example.org/source", "raw_content": "Available evidence"}]

    websocket = SimpleNamespace(send_json=AsyncMock())
    researcher = SimpleNamespace(
        retrievers=[TavilySearch, KnowledgeBase],
        cfg=SimpleNamespace(max_search_results_per_query=5),
        websocket=websocket, verbose=False, report_type="subtopic_report",
        role="Researcher", parent_query="", kwargs={}, add_costs=Mock(),
        visited_urls=set(), vector_store=None, add_research_sources=Mock(),
        scraper_manager=SimpleNamespace(browse_urls=AsyncMock(return_value=[])),
        context_manager=SimpleNamespace(
            get_similar_content_by_query=AsyncMock(return_value="Available evidence")
        ),
    )
    conductor = ResearchConductor(researcher)
    context = await conductor._get_context_by_web_search("topic")

    assert context == "Available evidence"
    assert request.call_count == 2
    assert [(e["stage"], e["code"], e["attempts"]) for e in conductor.search_errors] == [
        ("initial_search", "authentication", 1), ("sub_query_search", "authentication", 1)
    ]
    events = [call.args[0] for call in websocket.send_json.await_args_list]
    assert [event["metadata"] for event in events if event["content"] == "search_error"] == conductor.search_errors
    assert "internal response details" not in str(events)


async def test_mcp_tool_failure_is_distinct_from_successful_tool_evidence(monkeypatch):
    async def fail_tool():
        raise ToolException("internal MCP details")

    async def lookup_tool():
        return "Available evidence", {"structured_content": {"results": [
            {"url": "https://example.org/evidence", "content": "Available evidence"}
        ]}}

    tools = [
        StructuredTool.from_function(coroutine=fail_tool, name="fail", description="Fail", handle_tool_error=True),
        StructuredTool.from_function(coroutine=lookup_tool, name="lookup", description="Lookup",
                                     response_format="content_and_artifact"),
    ]
    response = SimpleNamespace(content="Model-only speculation", tool_calls=[
        {"id": "failed", "name": "fail", "args": {}},
        {"id": "success", "name": "lookup", "args": {}},
    ])
    model = SimpleNamespace(bind_tools=lambda tools: SimpleNamespace(ainvoke=AsyncMock(return_value=response)))
    monkeypatch.setattr(GenericLLMProvider, "from_provider", Mock(return_value=SimpleNamespace(llm=model)))
    skill = MCPResearchSkill(SimpleNamespace(strategic_llm_model="test", strategic_llm_provider="test", llm_kwargs={}))
    record = AsyncMock()

    results = await skill.conduct_research_with_tools("topic", tools, on_error=record)

    assert len(results) == 1
    assert results[0]["href"] == "https://example.org/evidence"
    error = record.await_args.args[0]
    assert (error.provider, error.code) == ("MCP/fail", "tool_error")
    assert "internal MCP details" not in str(error)


async def test_evidence_failure_stops_report_work_and_emits_reason():
    researcher = GPTResearcher.__new__(GPTResearcher)
    researcher._log_event = AsyncMock()
    researcher.query = "topic"
    researcher.report_type = "research_report"
    researcher.agent = researcher.role = "researcher"
    researcher.websocket = SimpleNamespace(send_json=AsyncMock())
    researcher.research_conductor = SimpleNamespace(search_errors=[], web_search_attempted=False)

    async def no_evidence():
        researcher.research_conductor.web_search_attempted = True
        return []

    researcher.research_conductor.conduct_research = no_evidence
    with pytest.raises(InsufficientSearchEvidence, match="Insufficient evidence"):
        await researcher.conduct_research()
    assert researcher.websocket.send_json.await_args.args[0]["content"] == "insufficient_evidence"


async def test_partial_report_displays_search_limitations_without_sending_them_to_writer():
    researcher = GPTResearcher.__new__(GPTResearcher)
    researcher._log_event = AsyncMock()
    researcher.available_images = []
    researcher.context = "Available evidence"
    researcher.research_conductor = SimpleNamespace(search_errors=[{
        "provider": "Tavily", "code": "timeout", "message": "The search request timed out.",
        "stage": "sub_query_search", "attempts": 3,
    }])
    writer = AsyncMock(return_value="Report with evidence.")
    researcher.report_generator = SimpleNamespace(write_report=writer)

    report = await researcher.write_report()

    assert writer.await_args.kwargs["ext_context"] == "Available evidence"
    assert "Report with evidence." in report
    assert "**Tavily** — The search request timed out." in report


@pytest.mark.parametrize("available_section", [True, False])
async def test_parallel_sections_preserve_failures_without_synthesizing_empty_research(monkeypatch, available_section):
    from multi_agents.agents.editor import EditorAgent
    from multi_agents.agents.publisher import PublisherAgent
    from multi_agents.agents.researcher import ResearchAgent

    failure = {"provider": "Tavily", "code": "timeout", "message": "The search request timed out."}

    async def research(self, query, **kwargs):
        if query == "Unavailable":
            raise InsufficientSearchEvidence([failure])
        return "Supported findings"

    monkeypatch.setattr(ResearchAgent, "research", research)
    state = {
        "task": {"query": "topic", "follow_guidelines": False, "verbose": False},
        "title": "topic",
        "sections": ["Unavailable", "Available"] if available_section else ["Unavailable"],
    }
    if not available_section:
        with pytest.raises(InsufficientSearchEvidence) as caught:
            await EditorAgent().run_parallel_research(state)
        assert caught.value.search_errors == [failure]
        return

    result = await EditorAgent().run_parallel_research(state)
    report = PublisherAgent(".").generate_layout(result)
    assert "Supported findings" in report
    assert "Insufficient evidence" in report
    assert "Tavily" in report


async def test_report_endpoint_returns_structured_evidence_failure(monkeypatch):
    from httpx import ASGITransport, AsyncClient
    from backend.server import app as api

    errors = [{"provider": "Tavily", "code": "timeout", "attempts": 3}]
    monkeypatch.setattr(api, "write_report", AsyncMock(side_effect=InsufficientSearchEvidence(errors)))
    async with AsyncClient(transport=ASGITransport(app=api.app), base_url="http://test") as client:
        response = await client.post("/report/", json={
            "task": "topic", "report_type": "research_report", "report_source": "web",
            "tone": "Objective", "repo_name": "", "branch_name": "", "generate_in_background": False,
        })
    assert response.status_code == 422
    assert response.json()["detail"]["search_errors"] == errors
