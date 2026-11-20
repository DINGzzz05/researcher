"""Claim-level citation auditing."""

from gpt_researcher.actions.markdown_processing import add_references
from gpt_researcher.citations import (
    extract_claims,
    normalize_url,
    verify_citations,
)

REPORT = """# Battery market outlook

The global battery market grew 12% in 2024 ([source](https://example.com/a)). Analysts expect growth to continue through 2026, driven by demand.

This paragraph makes a claim with no citation at all, and it is easily long enough to count.

## References

- [a](https://example.com/a)
- [b](https://example.com/b)
"""

ALLOWED = ["https://example.com/a", "https://example.com/b"]


def test_extract_claims_skips_headings_and_references():
    claims = extract_claims(REPORT)

    assert len(claims) == 3
    assert claims[0].sources == ("https://example.com/a",)
    assert claims[1].sources == ()
    assert all("References" not in claim.text for claim in claims)


def test_verify_citations_reports_uncited_and_unused_sources():
    audit = verify_citations(REPORT, ALLOWED)

    assert audit.total_claims == 3
    assert audit.cited_claims == 1
    assert audit.coverage == 1 / 3
    assert audit.uncited_claims == [claim.text for claim in audit.claims if not claim.sources]
    assert audit.cited_sources == ["https://example.com/a"]
    assert audit.unknown_sources == []
    assert audit.unused_sources == ["https://example.com/b"]


def test_verify_citations_flags_unknown_sources():
    report = REPORT.replace(
        "driven by demand.",
        "driven by demand ([other](https://evil.example/x)).",
    )

    audit = verify_citations(report, ALLOWED)

    assert audit.unknown_sources == ["https://evil.example/x"]


def test_verify_citations_ignores_fragments_and_trailing_slashes():
    report = (
        "The global battery market grew twelve percent in 2024 "
        "([source](https://example.com/a/#section))."
    )
    assert normalize_url("https://example.com/a/#section") == "https://example.com/a"

    audit = verify_citations(report, ["https://example.com/a/"])

    assert audit.unknown_sources == []
    assert audit.cited_sources == ["https://example.com/a/"]


def test_short_uncited_sentences_are_not_claims():
    assert extract_claims("The market grew.") == []


def test_url_with_parentheses_is_parsed_whole():
    report = (
        "The disambiguation page explains the term in detail "
        "([wiki](https://en.wikipedia.org/wiki/Foo_(bar)))."
    )

    claims = extract_claims(report)

    assert claims[0].sources == ("https://en.wikipedia.org/wiki/Foo_(bar)",)


def test_short_sentence_with_a_citation_is_a_claim():
    claims = extract_claims("The market grew ([a](https://example.com/a)).")

    assert len(claims) == 1
    assert claims[0].sources == ("https://example.com/a",)


def test_add_references_can_list_only_cited_sources():
    visited = {"https://example.com/a", "https://example.com/b"}
    report = REPORT.split("## References")[0]

    all_refs = add_references(report, visited)
    cited_refs = add_references(report, visited, only_cited=True)

    assert "https://example.com/b" in all_refs
    assert "https://example.com/b" not in cited_refs
    assert "https://example.com/a" in cited_refs


def test_add_references_returns_report_unchanged_without_citations():
    report = "This report cites nothing at all, which is long enough to be a claim."

    assert add_references(report, {"https://example.com/a"}, only_cited=True) == report
