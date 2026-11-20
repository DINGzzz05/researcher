"""Layered memory: episodes, facts, profile, consolidation and recall."""

import json
import sqlite3
from types import SimpleNamespace

import pytest

from gpt_researcher.memory.long_term import (
    ResearchMemory,
    extract_keywords,
    format_memory_block,
    memory_from_config,
)

REPORT = """# Battery Recycling in the EU

## Introduction

The EU battery regulation sets collection targets for member states.

## Conclusion

Lithium recovery now reaches 70% in pilot plants, but cobalt separation remains the bottleneck. Policy support is the main constraint on scaling.
"""

REPORT_UPDATED = """# Battery Recycling in the EU

## Conclusion

Lithium recovery now reaches 85% in pilot plants, but cobalt separation remains the bottleneck.
"""


@pytest.fixture()
def clock():
    return {"now": 1_000_000.0}


@pytest.fixture()
def memory(tmp_path, clock):
    return ResearchMemory(str(tmp_path / "memory.sqlite3"), clock=lambda: clock["now"])


def _run(memory, query="battery recycling", report=REPORT, **kwargs):
    kwargs.setdefault("sources", ["https://example.com/a"])
    return memory.remember_run(query, "Lithium recovery reaches 70%.", report=report, **kwargs)


# ------------------------------------------------------------------ write path


def test_a_run_writes_an_episode_and_its_facts(memory):
    write = _run(memory, session_id="s1")

    stats = memory.stats()
    assert stats["episodes"] == 1
    assert stats["facts"] >= 1
    assert write.episode.query == "battery recycling"
    assert write.episode.session_id == "s1"


def test_the_same_claim_merges_instead_of_duplicating(memory, clock):
    _run(memory)
    before = memory.stats()["facts"]

    clock["now"] += 3_600
    _run(memory)

    assert memory.stats()["facts"] == before
    fact = memory.recall("lithium recovery").facts[0]
    assert fact.times_seen == 2
    assert fact.confidence > 0.55


def test_a_changed_number_supersedes_the_old_fact(memory, clock):
    _run(memory)
    clock["now"] += 86_400
    _run(memory, report=REPORT_UPDATED)

    stats = memory.stats()
    assert stats["superseded"] == 1
    claims = [fact.text for fact in memory.recall("lithium recovery").facts]
    assert any("85%" in claim for claim in claims)
    assert not any("70%" in claim for claim in claims)


def test_superseded_facts_are_kept_as_history(memory, clock):
    _run(memory)
    clock["now"] += 86_400
    _run(memory, report=REPORT_UPDATED)

    with sqlite3.connect(memory.path) as conn:
        rows = conn.execute(
            "SELECT claim, valid_to FROM facts WHERE valid_to IS NOT NULL"
        ).fetchall()
    assert rows and "70%" in rows[0][0] and rows[0][1] is not None


# ------------------------------------------------------------------- recall


def test_recall_splits_the_layers(memory):
    _run(memory, session_id="s1")

    context = memory.recall("lithium recovery", session_id="s1")

    assert context.session and not context.episodes
    assert context.facts
    assert context.profile


def test_another_session_is_recalled_as_an_episode(memory):
    _run(memory, session_id="s1")

    context = memory.recall("battery recycling", session_id="s2")

    assert context.session == []
    assert context.episodes


def test_facts_are_recalled_by_entity_even_without_word_overlap(memory):
    memory.remember_run(
        "vLLM scheduling",
        "The engine batches requests continuously to keep the GPU busy.",
        report=(
            "# vLLM\n\n## Conclusion\n\nThe engine batches requests continuously "
            "to keep the GPU busy and raises throughput."
        ),
        sources=["https://example.com/vllm"],
    )

    context = memory.recall("vLLM")

    assert any("batches requests" in fact.text for fact in context.facts)


def test_unrelated_queries_recall_nothing(memory):
    _run(memory)

    context = memory.recall("sourdough starter hydration")

    assert context.is_empty()


def test_a_follow_up_without_keywords_keeps_the_last_run_in_view(memory):
    _run(memory, session_id="s1")

    context = memory.recall("What about its latency?", session_id="s1")

    assert len(context.session) == 1
    assert context.episodes == []
    assert context.facts  # the working set brings its own facts along


def test_switching_topics_does_not_drag_the_last_session_run_along(memory):
    _run(memory, session_id="s1")
    memory.remember_run(
        "vLLM scheduling",
        "Batching raises throughput on GPU.",
        report="# vLLM\n\n## Conclusion\n\nBatching raises throughput on GPU by 2x.",
        sources=["https://example.com/v"],
        session_id="s1",
    )

    context = memory.recall("lithium recovery", session_id="s1")

    assert all(hit.query != "vLLM scheduling" for hit in context.session)


def test_expired_episodes_go_before_facts(tmp_path, clock):
    memory = ResearchMemory(
        str(tmp_path / "m.sqlite3"),
        episode_ttl_days=1,
        fact_ttl_days=30,
        clock=lambda: clock["now"],
    )
    _run(memory, session_id="s1")

    clock["now"] += 2 * 86_400
    context = memory.recall("lithium recovery")

    assert context.session == [] and context.episodes == []
    assert context.facts


def test_users_are_isolated(memory):
    _run(memory, user_id="u1")

    assert memory.recall("battery recycling", user_id="u2").is_empty()
    assert not memory.recall("battery recycling", user_id="u1").is_empty()


def test_profile_tracks_recurring_topics(memory):
    _run(memory, query="battery recycling policy")
    _run(memory, query="battery recycling capacity")
    _run(memory, query="solar panel efficiency")

    prefs = memory.preferences(top=4)

    assert {"battery", "recycling"}.issubset(set(prefs))


# ----------------------------------------------------------------- lifecycle


def test_forget_expired_removes_only_expired_rows(tmp_path, clock):
    memory = ResearchMemory(
        str(tmp_path / "m.sqlite3"),
        episode_ttl_days=1,
        fact_ttl_days=30,
        clock=lambda: clock["now"],
    )
    _run(memory)

    clock["now"] += 2 * 86_400
    removed = memory.forget_expired()

    assert removed == 1
    assert memory.stats()["episodes"] == 0
    assert memory.stats()["facts"] >= 1


def test_forget_clears_the_user(memory):
    _run(memory, user_id="u1")

    memory.forget(user_id="u1")

    assert memory.stats(user_id="u1") == {
        "episodes": 0, "facts": 0, "superseded": 0, "profile_terms": 0
    }


def test_legacy_memories_table_is_migrated(tmp_path):
    path = str(tmp_path / "legacy.sqlite3")
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE memories (
                id TEXT PRIMARY KEY, tier TEXT, user_id TEXT, session_id TEXT,
                query TEXT, summary TEXT, sources TEXT, keywords TEXT,
                created_at REAL, expires_at REAL
            );
            """
        )
        conn.execute(
            "INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                "old-1", "long", "u1", None, "battery recycling",
                "Lithium recovery now reaches 70% in pilot plants.",
                json.dumps(["https://example.com/a"]), json.dumps(["battery"]),
                1_000.0, 2_000.0,
            ),
        )

    memory = ResearchMemory(path, clock=lambda: 1_000.0)
    stats = memory.stats()

    assert stats["episodes"] == 1
    assert memory.recall("battery recycling", user_id="u1").episodes
    with sqlite3.connect(path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
    assert "memories_legacy_v1" in tables


# ----------------------------------------------------------------- rendering


def test_memory_block_is_empty_without_a_context():
    assert format_memory_block(None) == ""


def test_memory_block_renders_the_layers_within_budget(memory):
    _run(memory, session_id="s1")

    block = format_memory_block(memory.recall("lithium recovery", session_id="s1"), max_chars=600)

    assert "Prior research memory" in block
    assert "### This session" in block
    assert "### Established facts" in block
    assert "Focus:" in block
    assert len(block) <= 600


def test_agent_prepends_memory_to_the_context(tmp_path):
    store = ResearchMemory(str(tmp_path / "m.sqlite3"))
    store.remember_run(
        "battery cost", "Lithium prices fell.", report=REPORT, user_id="u1",
    )
    agent = _bare_agent(tmp_path)

    context = agent._with_memory_context("SCRAPED CONTEXT")

    assert "Prior research memory" in context
    assert context.endswith("SCRAPED CONTEXT")


def test_agent_leaves_context_alone_when_memory_is_off(tmp_path):
    agent = _bare_agent(tmp_path, enabled=False)

    assert agent._with_memory_context("SCRAPED CONTEXT") == "SCRAPED CONTEXT"


# ---------------------------------------------------------------- agent wiring


def _bare_agent(tmp_path, *, enabled=True, user_id="u1"):
    from gpt_researcher.agent import GPTResearcher

    agent = GPTResearcher.__new__(GPTResearcher)
    agent.cfg = SimpleNamespace(
        long_term_memory=enabled,
        memory_db_path=str(tmp_path / "m.sqlite3"),
        memory_ttl_days=30,
        memory_fact_ttl_days=365,
        memory_session_ttl_hours=24,
        memory_max_records=100,
        memory_max_facts=100,
        memory_facts_per_run=6,
        memory_top_k=5,
        memory_user_id=user_id,
        memory_summary_chars=800,
        memory_max_sources=12,
    )
    agent.query = "battery recycling"
    agent.memory_user_id = user_id
    agent.memory_session_id = "s1"
    agent.memory_summary = None
    agent._long_term_memory = None
    return agent


def test_agent_stores_episode_facts_and_sources(tmp_path):
    agent = _bare_agent(tmp_path)
    agent.get_source_urls = lambda: [
        "https://www.example.com/a?utm_source=x#top",
        "mailto:hi@example.com",
    ]

    agent._remember_report(REPORT)

    store = ResearchMemory(str(tmp_path / "m.sqlite3"))
    assert store.stats(user_id="u1")["episodes"] == 1
    context = store.recall("lithium recovery", user_id="u1", session_id="s1")
    assert context.session[0].sources == ("https://example.com/a",)
    assert context.session[0].text.startswith("Lithium recovery now reaches 70%")
    assert any("Lithium recovery" in fact.text for fact in context.facts)


def test_agent_uses_the_summary_the_writer_selected(tmp_path):
    agent = _bare_agent(tmp_path)
    agent.get_source_urls = lambda: []
    agent.memory_summary = "WRITER SELECTED SUMMARY"

    agent._remember_report(REPORT)

    store = ResearchMemory(str(tmp_path / "m.sqlite3"))
    assert store.recall("battery", user_id="u1").episodes[0].text == "WRITER SELECTED SUMMARY"
    assert agent.memory_summary is None


def test_memory_from_config_is_off_by_default():
    assert memory_from_config(SimpleNamespace(long_term_memory=False)) is None


def test_memory_from_config_maps_the_layers(tmp_path):
    cfg = SimpleNamespace(
        long_term_memory=True,
        memory_db_path=str(tmp_path / "m.sqlite3"),
        memory_ttl_days=7,
        memory_fact_ttl_days=90,
        memory_session_ttl_hours=2,
        memory_max_records=10,
        memory_max_facts=20,
        memory_facts_per_run=3,
    )

    store = memory_from_config(cfg)

    assert store is not None
    assert store.episode_ttl == 7 * 86_400
    assert store.fact_ttl == 90 * 86_400
    assert store.max_episodes == 10 and store.max_facts == 20
