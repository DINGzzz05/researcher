"""The citation gate must pass its committed baseline and fail on growth."""

import json
import shutil
from pathlib import Path

from evals.citation_gate.gate import audit_reports, compare, main

GATE_DIR = Path(__file__).resolve().parents[1] / "evals" / "citation_gate"
FIXTURES = GATE_DIR / "reports"
BASELINE = GATE_DIR / "baseline.json"


def _baseline() -> dict:
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def _copy_fixtures(target: Path) -> None:
    for path in FIXTURES.iterdir():
        shutil.copy(path, target / path.name)


def test_gate_matches_the_committed_baseline():
    assert compare(audit_reports(FIXTURES), _baseline()) == []


def test_gate_flags_a_new_ghost_url(tmp_path):
    _copy_fixtures(tmp_path)
    (tmp_path / "new-ghost.md").write_text(
        "# New section\n\n"
        "A claim long enough to be counted as substantive cites a source that was "
        "never retrieved during the run at all ([x](https://never-retrieved.example/a)).\n",
        encoding="utf-8",
    )
    (tmp_path / "new-ghost.sources.json").write_text("[]", encoding="utf-8")

    problems = compare(audit_reports(tmp_path), _baseline())
    assert any("ghost URLs" in problem for problem in problems)


def test_gate_flags_growing_uncited_claims(tmp_path):
    _copy_fixtures(tmp_path)
    (tmp_path / "more-uncited.md").write_text(
        "# Another section\n\n"
        "This paragraph states a substantive claim without attaching any citation "
        "to it, so the audit should count it as uncited.\n",
        encoding="utf-8",
    )
    (tmp_path / "more-uncited.sources.json").write_text("[]", encoding="utf-8")

    problems = compare(audit_reports(tmp_path), _baseline())
    assert any("uncited claims" in problem for problem in problems)


def test_reports_without_sources_are_skipped(tmp_path):
    _copy_fixtures(tmp_path)
    (tmp_path / "no-sources.md").write_text("# No sources\n\nSome text.\n", encoding="utf-8")

    current = audit_reports(tmp_path)
    assert current["skipped"] == 1
    assert current["reports"] == 3


def test_cli_passes_and_fails(tmp_path):
    assert main(["--reports", str(FIXTURES), "--baseline", str(BASELINE)]) == 0

    _copy_fixtures(tmp_path)
    (tmp_path / "ghost.md").write_text(
        "# Ghost\n\n"
        "A substantive claim citing a URL that was never retrieved by the run at any point "
        "([x](https://ghost.example/z)).\n",
        encoding="utf-8",
    )
    (tmp_path / "ghost.sources.json").write_text("[]", encoding="utf-8")
    assert main(["--reports", str(tmp_path), "--baseline", str(BASELINE)]) == 1
