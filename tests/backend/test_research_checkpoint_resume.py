import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from multi_agents.agents.orchestrator import ChiefEditorAgent
from multi_agents.memory.research import ResearchState


@pytest.mark.asyncio
async def test_resume_reuses_saved_research_and_completed_report(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    calls = []

    async def research(state):
        calls.append("research")
        return {"initial_research": "saved evidence"}

    async def writer(state):
        calls.append("writer")
        if calls.count("writer") == 1:
            raise TimeoutError("model timed out")
        return {"report": state["initial_research"]}

    def build_graph(self):
        graph = StateGraph(ResearchState)
        graph.add_node("research", research)
        graph.add_node("writer", writer)
        graph.add_edge(START, "research")
        graph.add_edge("research", "writer")
        graph.add_edge("writer", END)
        return graph

    monkeypatch.setattr(ChiefEditorAgent, "init_research_team", build_graph)
    saver = InMemorySaver()
    task = {"query": "test research"}

    with pytest.raises(TimeoutError, match="model timed out"):
        await ChiefEditorAgent(task, task_id="test-run").run_research_task(checkpointer=saver)

    for _ in range(2):
        result = await ChiefEditorAgent(task, task_id="test-run").run_research_task(
            checkpointer=saver, resume=True,
        )
        assert result["report"] == "saved evidence"

    assert calls == ["research", "writer", "writer"]
