"""Citation-audit regression gate.

``gpt_researcher.citations`` audits a finished report without an LLM: it maps
claims to their cited sources and flags URLs the run never retrieved. That
audit is useful one report at a time, but the number that matters for a
regression gate is whether *ghost URLs* or *uncited claims* grow over time -
for example when a prompt change makes the writer cite from memory again.

This module aggregates the audit over a directory of saved reports and
compares the totals with a committed baseline. Every report is a ``*.md`` file
paired with a ``*.sources.json`` list of the URLs its run retrieved; reports
without that file are skipped, because an audit without the allowed-URL set
cannot tell a ghost source from an unlisted one.

Run it in CI (``.github/workflows/citation-gate.yml``) or by hand::

    python -m evals.citation_gate.gate --reports evals/citation_gate/reports \\
        --baseline evals/citation_gate/baseline.json

Exit code 1 means a metric regressed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gpt_researcher.citations import verify_citations

#: Metrics reported for every run; keep in sync with ``baseline.json``.
METRICS = ("reports", "claims", "cited_claims", "uncited_claims", "unknown_sources", "coverage")


def audit_reports(reports_dir: Path) -> dict:
    """Aggregate the citation audit over every report with a sources file."""
    totals = {
        "reports": 0,
        "skipped": 0,
        "claims": 0,
        "cited_claims": 0,
        "uncited_claims": 0,
        "unknown_sources": 0,
        "coverage": 0.0,
        "per_report": [],
    }

    for report_path in sorted(reports_dir.glob("*.md")):
        sources_path = report_path.with_suffix(".sources.json")
        if not sources_path.exists():
            totals["skipped"] += 1
            continue

        allowed = json.loads(sources_path.read_text(encoding="utf-8"))
        audit = verify_citations(report_path.read_text(encoding="utf-8"), allowed)

        totals["reports"] += 1
        totals["claims"] += audit.total_claims
        totals["cited_claims"] += audit.cited_claims
        totals["uncited_claims"] += len(audit.uncited_claims)
        totals["unknown_sources"] += len(audit.unknown_sources)
        totals["per_report"].append(
            {
                "report": report_path.name,
                "claims": audit.total_claims,
                "uncited_claims": len(audit.uncited_claims),
                "unknown_sources": len(audit.unknown_sources),
            }
        )

    if totals["claims"]:
        totals["coverage"] = round(totals["cited_claims"] / totals["claims"], 3)
    return totals


def compare(
    current: dict,
    baseline: dict,
    *,
    max_unknown_regression: int = 0,
    max_coverage_drop: float = 0.0,
    max_uncited_regression: int = 0,
) -> list[str]:
    """Human-readable regressions of ``current`` against ``baseline`` (empty = pass)."""
    problems: list[str] = []

    if current["unknown_sources"] > baseline["unknown_sources"] + max_unknown_regression:
        problems.append(
            f"ghost URLs: {current['unknown_sources']} > baseline {baseline['unknown_sources']}"
            f" (allowed +{max_unknown_regression})"
        )
    if current["uncited_claims"] > baseline["uncited_claims"] + max_uncited_regression:
        problems.append(
            f"uncited claims: {current['uncited_claims']} > baseline {baseline['uncited_claims']}"
            f" (allowed +{max_uncited_regression})"
        )
    if current["coverage"] + max_coverage_drop < baseline["coverage"]:
        problems.append(
            f"coverage: {current['coverage']:.3f} < baseline {baseline['coverage']:.3f}"
            f" (allowed -{max_coverage_drop})"
        )
    return problems


def _print_summary(current: dict) -> None:
    print(f"reports audited: {current['reports']} (skipped {current['skipped']})")
    print(f"claims: {current['claims']}  cited: {current['cited_claims']}  "
          f"uncited: {current['uncited_claims']}")
    print(f"ghost URLs: {current['unknown_sources']}  coverage: {current['coverage']:.3f}")
    for row in current["per_report"]:
        print(
            f"  {row['report']}: claims={row['claims']}"
            f" uncited={row['uncited_claims']} ghost={row['unknown_sources']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Citation-audit regression gate.")
    parser.add_argument("--reports", type=Path, default=Path("evals/citation_gate/reports"))
    parser.add_argument("--baseline", type=Path, default=Path("evals/citation_gate/baseline.json"))
    parser.add_argument("--max-unknown-regression", type=int, default=0)
    parser.add_argument("--max-uncited-regression", type=int, default=0)
    parser.add_argument("--max-coverage-drop", type=float, default=0.0)
    parser.add_argument("--write-baseline", action="store_true",
                        help="overwrite the baseline with the current totals and exit 0")
    args = parser.parse_args(argv)

    if not args.reports.is_dir():
        print(f"reports directory not found: {args.reports}", file=sys.stderr)
        return 2

    current = audit_reports(args.reports)
    _print_summary(current)

    if args.write_baseline:
        args.baseline.write_text(
            json.dumps({key: current[key] for key in METRICS}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"baseline written: {args.baseline}")
        return 0

    if not args.baseline.exists():
        print(f"baseline not found: {args.baseline} (run with --write-baseline)", file=sys.stderr)
        return 2

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    problems = compare(
        current,
        baseline,
        max_unknown_regression=args.max_unknown_regression,
        max_uncited_regression=args.max_uncited_regression,
        max_coverage_drop=args.max_coverage_drop,
    )
    if problems:
        print("\nREGRESSION:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print("\nOK: no citation regressions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
