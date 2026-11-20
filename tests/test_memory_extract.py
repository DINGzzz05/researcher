"""Report -> memory record: summary selection and source cleaning."""

from types import SimpleNamespace

from gpt_researcher.memory.extract import (
    clean_sources,
    extract_entities,
    extract_facts,
    select_summary,
)

REPORT = """# Battery Recycling in the EU

Recycling capacity doubled between 2019 and 2024, driven by regulation.

## Introduction

The EU battery regulation sets collection targets for member states.

## Costs

Collection costs vary widely by country and by chemistry.

## Conclusion

Lithium recovery now reaches 70% in pilot plants, but cobalt separation
remains the bottleneck. Policy support is the main constraint on scaling.
"""


def test_summary_leads_with_the_conclusion_not_the_title():
    summary = select_summary("battery recycling", REPORT)

    assert summary.startswith("Lithium recovery now reaches 70%")
    assert "Battery Recycling in the EU" not in summary


def test_summary_strips_markdown_syntax():
    report = (
        "# Title\n\n"
        "## Conclusion\n\n"
        "**Cobalt** separation is the bottleneck, see [the study](https://example.com/s)."
    )

    summary = select_summary("cobalt", report)

    assert summary == "Cobalt separation is the bottleneck, see the study."


def test_summary_falls_back_to_the_lead_without_a_conclusion():
    report = (
        "# Solar Costs\n\n"
        "Solar module prices fell 40% in two years. Payback periods shortened.\n\n"
        "## Drivers\n\n"
        "Manufacturing scale is the main driver."
    )

    summary = select_summary("solar costs", report)

    assert summary.startswith("Solar module prices fell 40%")


def test_summary_is_cut_at_a_sentence_boundary():
    report = "# T\n\n## Conclusion\n\n" + ("Lithium recovery improves. " * 80)

    summary = select_summary("lithium", report, limit=120)

    assert len(summary) <= 121
    assert summary.endswith(".")


def test_summary_prefers_the_parts_the_writer_assembled():
    summary = select_summary(
        "battery",
        parts=[
            ("introduction", "This report covers EU recycling."),
            ("body", "Collection costs vary by country."),
            ("conclusion", "Cobalt separation is the bottleneck."),
        ],
    )

    assert summary.startswith("Cobalt separation is the bottleneck.")


def test_summary_uses_the_body_when_conclusion_is_missing():
    summary = select_summary(
        "cobalt",
        parts=[
            ("introduction", "Scope."),
            ("body", "Unrelated filler.\n\nCobalt separation is the bottleneck."),
        ],
    )

    assert "Cobalt separation is the bottleneck." in summary


def test_clean_sources_canonicalises_and_dedupes():
    urls = [
        "https://www.Example.com/a?utm_source=x#frag",
        "https://example.com/a/",
        "http://example.com/a",
        "https://example.com/b?id=2&utm_medium=y",
    ]

    assert clean_sources(urls) == [
        "https://example.com/a",
        "https://example.com/b?id=2",
    ]


def test_clean_sources_drops_non_http_entries():
    urls = ["mailto:hi@example.com", "javascript:void(0)", "ftp://example.com/f", ""]

    assert clean_sources(urls) == []


def test_clean_sources_unwraps_search_redirects():
    urls = [
        "https://www.google.com/url?q=https://example.org/page&sa=D",
        "https://www.bing.com/ck/a?!&&u=a1aHR0cHM6Ly9leGFtcGxlLm5ldC94",
    ]

    assert clean_sources(urls) == ["https://example.org/page", "https://example.net/x"]


def test_clean_sources_drops_search_result_pages():
    urls = ["https://www.google.com/search?q=battery", "https://duckduckgo.com/?q=battery"]

    assert clean_sources(urls) == []


def test_clean_sources_caps_each_domain():
    urls = [f"https://example.com/page-{i}" for i in range(5)]

    assert clean_sources(urls, per_domain=2) == [
        "https://example.com/page-0",
        "https://example.com/page-1",
    ]


def test_clean_sources_puts_query_matches_first():
    urls = ["https://example.com/solar-panels", "https://example.com/battery-recycling"]

    assert clean_sources(urls, query="battery recycling")[0] == (
        "https://example.com/battery-recycling"
    )


def test_clean_sources_respects_the_limit():
    urls = [f"https://site-{i}.com/x" for i in range(20)]

    assert len(clean_sources(urls, limit=3)) == 3


# ----------------------------------------------------------------------- facts


def test_facts_come_from_the_conclusion():
    facts = extract_facts(REPORT, "battery recycling", limit=5)

    assert any("70%" in fact for fact in facts)
    assert any("Policy support" in fact for fact in facts)
    assert not any("Battery Recycling in the EU" in fact for fact in facts)


def test_facts_skip_report_meta_talk():
    report = (
        "# Title\n\n## Conclusion\n\n"
        "This report covers the EU market and the policy changes that shaped it in 2024."
    )

    assert extract_facts(report) == []


def test_facts_ignore_the_introduction_when_a_conclusion_exists():
    report = (
        "# Title\n\n## Introduction\n\nThe market has grown quickly.\n\n"
        "## Conclusion\n\nCosts fell 30% in 2024."
    )

    assert extract_facts(report) == ["Costs fell 30% in 2024."]


def test_claims_drop_leading_transitions_and_keep_the_capital():
    report = "# Title\n\n## Conclusion\n\nHowever, cobalt separation still limits yields at scale."

    assert extract_facts(report) == ["Cobalt separation still limits yields at scale."]


def test_facts_are_deduplicated_and_capped():
    report = "# Title\n\n## Conclusion\n\n" + "Costs fell 30% in 2024. " * 10

    facts = extract_facts(report, limit=3)

    assert len(facts) == 1


def test_entities_keep_acronyms_and_camel_case():
    entities = extract_entities("vLLM and SGLang benchmark on GPU, see 《EU AI Act》")

    assert {"vLLM", "SGLang", "GPU"}.issubset(set(entities))
    assert "EU AI Act" in entities


def _writer_researcher(**cfg_overrides):
    cfg = SimpleNamespace(
        agent_role="", citation_check=False,
        long_term_memory=True, memory_summary_chars=800,
    )
    cfg.__dict__.update(cfg_overrides)
    return SimpleNamespace(
        query="battery recycling",
        context=["scraped context"],
        report_type="research_report",
        report_source="web",
        tone="",
        websocket=None,
        headers={},
        kwargs={},
        role="",
        verbose=False,
        cfg=cfg,
        memory_summary=None,
        get_research_images=lambda: [],
        add_costs=lambda *args, **kwargs: None,
    )


async def test_writer_selects_the_summary_while_it_assembles_the_report(monkeypatch):
    from gpt_researcher.skills import writer as writer_module

    async def fake_generate_report(**kwargs):
        return "# Battery Recycling\n\n## Conclusion\n\nCobalt is the bottleneck."

    monkeypatch.setattr(writer_module, "generate_report", fake_generate_report)
    researcher = _writer_researcher()

    await writer_module.ReportGenerator(researcher).write_report()

    assert researcher.memory_summary == "Cobalt is the bottleneck."


async def test_writer_leaves_the_summary_alone_when_memory_is_off(monkeypatch):
    from gpt_researcher.skills import writer as writer_module

    async def fake_generate_report(**kwargs):
        return "# Battery Recycling\n\n## Conclusion\n\nCobalt is the bottleneck."

    monkeypatch.setattr(writer_module, "generate_report", fake_generate_report)
    researcher = _writer_researcher(long_term_memory=False)

    await writer_module.ReportGenerator(researcher).write_report()

    assert researcher.memory_summary is None
