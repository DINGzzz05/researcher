"""Section drafts: full text on disk, summaries in the graph state."""

from multi_agents.agents.claim_review import UNVERIFIED_CLAIM
from multi_agents.agents.drafts import persist_draft, read_draft, summary_of
from multi_agents.agents.publisher import PublisherAgent
from multi_agents.agents.writer import WriterAgent

HEADERS = {
    "title": "Title",
    "date": "Date",
    "introduction": "Introduction",
    "table_of_contents": "Contents",
    "conclusion": "Conclusion",
    "references": "References",
}


def _state(tmp_path, summary, name, *, unresolved=()):
    return {
        "research_data": [{"Market size": summary}],
        "section_drafts_dir": str(tmp_path),
        "section_drafts": {"Market size": name},
        "headers": HEADERS,
        "date": "2026-01-01",
        "introduction": "Intro",
        "table_of_contents": "TOC",
        "conclusion": "Conclusion",
        "sources": [],
        "unresolved_claim_sections": list(unresolved),
    }


def test_persist_keeps_full_body_on_disk_and_returns_a_summary(tmp_path):
    body = "## Market size\n\n" + ("Full body sentence. " * 100)
    summary, name = persist_draft(str(tmp_path), 0, "Market size", body)

    assert len(summary) < len(body)
    assert summary.startswith("## Market size")
    assert read_draft(str(tmp_path), name).rstrip("\n") == body.strip()


def test_summary_is_empty_safe_and_bounded():
    assert summary_of("") == ""
    assert len(summary_of("word " * 500)) <= 600 + 2


def test_publisher_publishes_the_full_draft_not_the_summary(tmp_path):
    body = "## Market size\n\nUNIQUE_FULL_BODY_MARKER the market grew."
    summary, name = persist_draft(str(tmp_path), 0, "Market size", body)

    layout = PublisherAgent(str(tmp_path)).generate_layout(_state(tmp_path, summary, name))

    assert "UNIQUE_FULL_BODY_MARKER" in layout
    assert summary not in layout or summary == body


def test_publisher_falls_back_to_the_state_body_when_the_file_is_gone(tmp_path):
    state = _state(tmp_path, "summary only", "missing.md")
    layout = PublisherAgent(str(tmp_path)).generate_layout(state)
    assert "summary only" in layout


def test_publisher_marks_unresolved_claims_in_disk_drafts(tmp_path):
    body = f"## Market size\n\nEvidence. {UNVERIFIED_CLAIM} More text."
    summary, name = persist_draft(str(tmp_path), 0, "Market size", body)

    layout = PublisherAgent(str(tmp_path)).generate_layout(_state(tmp_path, summary, name, unresolved=[0]))

    assert "**[证据不足" in layout


def test_writer_reads_bounded_draft_excerpts(tmp_path):
    body = "## Market size\n\nUNIQUE_EXCERPT_MARKER the market grew."
    summary, name = persist_draft(str(tmp_path), 0, "Market size", body)

    blob = WriterAgent()._research_blob(_state(tmp_path, summary, name))

    assert "UNIQUE_EXCERPT_MARKER" in blob


def test_writer_falls_back_to_state_summaries_without_drafts():
    state = {"research_data": [{"Market size": "state summary"}], "section_drafts": {}}
    assert "state summary" in WriterAgent()._research_blob(state)


async def test_parallel_research_persists_drafts_and_hands_back_summaries(tmp_path, monkeypatch):
    """The state carries summaries; each full draft is on disk and publisher reads it."""
    from multi_agents.agents.editor import EditorAgent

    sections = ["Market size", "Competition"]

    def body_for(topic):
        return f"## {topic}\n\n" + ("FULL " + topic + " body sentence. " * 120)

    class _FakeChain:
        async def ainvoke(self, payload, config=None):
            topic = payload["topic"]
            return {"draft": {topic: body_for(topic)}}

    class _FakeWorkflow:
        def compile(self):
            return _FakeChain()

    editor = EditorAgent()
    monkeypatch.setattr(editor, "_create_workflow", lambda: _FakeWorkflow())

    state = {
        "task": {"query": "battery market"},
        "sections": sections,
        "title": "Battery market",
        "section_drafts_dir": str(tmp_path / "sections"),
    }
    out = await editor.run_parallel_research(state)

    assert set(out["section_drafts"]) == set(sections)
    for topic in sections:
        summary = out["research_data"][sections.index(topic)][topic]
        assert len(summary) < len(body_for(topic))  # state keeps a summary only
        saved = read_draft(state["section_drafts_dir"], out["section_drafts"][topic])
        assert body_for(topic).strip() in saved  # disk keeps the full draft

    layout = PublisherAgent(str(tmp_path)).generate_layout(
        {
            **out,
            "section_drafts_dir": state["section_drafts_dir"],
            "headers": HEADERS,
            "date": "2026-01-01",
            "introduction": "Intro",
            "table_of_contents": "TOC",
            "conclusion": "Conclusion",
            "sources": [],
        }
    )
    for topic in sections:
        assert f"FULL {topic} body sentence." in layout
