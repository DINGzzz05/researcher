"""Reproducible offline checks, or context replay on saved real research pages.

Offline mode is a local documentation workload, not DeepResearch Bench and
not evidence of live report quality. No provider/LLM calls are made by it.

    python -m evals.researchpilot_metrics --offline --out outputs/metrics-local
    python -m evals.researchpilot_metrics --runs evals/context_filter/runs --out outputs/metrics-pages

--summarize is opt-in and uses the configured real LLM; it is incompatible
with --offline. This program does not estimate or fabricate FACT scores.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import socket
import statistics
import subprocess
import sys
import time
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "3582a6841fe7dc801e2a34e34ee408641aea9bce"
QUERIES = [
    "How does GPT Researcher plan and run deep research?",
    "How are research agents orchestrated in LangGraph?",
    "How can GPT Researcher research local documents and hybrid sources?",
    "How do context filtering BM25 keyword and embeddings work?",
    "How can GPT Researcher use MCP tools and servers?",
    "How do you configure retrievers and search engines?",
    "How do you configure LLM models and embeddings?",
    "How do you deploy GPT Researcher using Docker?",
    "How does the FastAPI backend stream research progress?",
    "How do you customize the research report and prompts?",
    "How do you use vector stores and Azure storage?",
    "How are research quality and citations evaluated?",
]


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], encoding="utf-8"
    )


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def install_offline_guard() -> None:
    """Fail instead of accidentally invoking a remote model during offline replay."""
    real_connect = socket.socket.connect

    def connect(sock, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if host not in {"localhost", "127.0.0.1", "::1"}:
            raise RuntimeError("Network disabled for the offline benchmark")
        return real_connect(sock, address, *args, **kwargs)

    socket.socket.connect = connect


def baseline_selector():
    # These are the direct shared imports of the upstream selector. Refuse to
    # describe this as an exact baseline if the dependencies have been edited.
    dependencies = [
        "gpt_researcher/context/lexical.py", "gpt_researcher/context/retriever.py",
        "gpt_researcher/prompts.py",
    ]
    for relative in dependencies:
        old = git("show", f"{BASELINE}:{relative}").replace("\r\n", "\n")
        current = (ROOT / relative).read_text(encoding="utf-8").replace("\r\n", "\n")
        if old != current:
            raise RuntimeError(f"Shared baseline dependency changed: {relative}")
    name = "gpt_researcher.context._metrics_upstream_select"
    module = types.ModuleType(name)
    module.__package__ = "gpt_researcher.context"
    sys.modules[name] = module
    source = git("show", f"{BASELINE}:gpt_researcher/context/select.py")
    exec(compile(source, f"{BASELINE}/context/select.py", "exec"), module.__dict__)
    return module.select_context


def documentation_runs() -> tuple[list[dict], list[dict]]:
    paths = git("ls-tree", "-r", "--name-only", BASELINE, "docs/docs/gpt-researcher").splitlines()
    pages, corpus = [], []
    for relative in paths:
        if not relative.endswith((".md", ".mdx")):
            continue
        text = git("show", f"{BASELINE}:{relative}")
        url = f"https://github.com/assafelovic/gpt-researcher/blob/{BASELINE}/{relative}"
        pages.append({"url": url, "title": Path(relative).stem, "raw_content": text})
        corpus.append({"path": relative, "sha256": hashlib.sha256(text.encode()).hexdigest()})
    return [
        {"id": f"docs-{i:02}", "query": query, "calls": [{"query": query, "pages": pages}]}
        for i, query in enumerate(QUERIES, 1)
    ], corpus


async def context_replay(args, out: Path) -> dict:
    import tiktoken
    from gpt_researcher.context.select import select_context

    encoder = tiktoken.get_encoding("o200k_base")
    upstream = baseline_selector()
    if args.offline:
        runs, corpus = documentation_runs()
        dump(out / "corpus_manifest.json", corpus)
        install_offline_guard()
        cfg = SimpleNamespace(
            context_filter="keyword", context_dedup=True, context_dedup_threshold=0.9,
            context_summarize=False, context_token_budget=12000,
            smart_llm_model="gpt-5.4", context_order="relevance",
        )
    else:
        from dotenv import load_dotenv
        from gpt_researcher.config import Config
        load_dotenv(ROOT / ".env")
        cfg = Config()
        cfg.context_filter = "keyword"  # isolate post-selection improvements
        cfg.context_summarize = args.summarize
        runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(args.runs.glob("*.json"))]
    if not runs or any(not r.get("calls") for r in runs):
        raise ValueError("Expected non-empty recorded research calls for every run")

    rows = []
    for run in runs:
        rendered = {}
        costs = []
        seconds = {}
        for label, selector in [("upstream", upstream), ("project", select_context)]:
            started = time.perf_counter()
            texts = []
            for call in run["calls"]:
                texts.append(await selector(
                    call["query"], call["pages"], cfg, max_results=10,
                    cost_callback=costs.append if label == "project" else None,
                ))
            rendered[label] = " ".join(t for t in texts if t)
            seconds[label] = time.perf_counter() - started
        tokens = {label: len(encoder.encode(text, disallowed_special=())) for label, text in rendered.items()}
        row = {
            "id": run["id"], "query": run["query"], "calls": len(run["calls"]),
            "upstream_tokens": tokens["upstream"], "project_tokens": tokens["project"],
            "reduction_pct": 100 * (1 - tokens["project"] / tokens["upstream"]) if tokens["upstream"] else None,
            "project_filter_cost_usd": sum(costs), "seconds": seconds,
        }
        rows.append(row)
        dump(out / "contexts" / f"{run['id']}.json", {**row, **rendered})
        print(json.dumps(row, ensure_ascii=True), flush=True)
    dump(out / "context_rows.json", rows)
    a = sum(r["upstream_tokens"] for r in rows)
    b = sum(r["project_tokens"] for r in rows)
    result = {
        "workload": "12 hand-written queries over frozen upstream repository documentation" if args.offline else "recorded research pages",
        "n_tasks": len(rows), "tokenizer": "o200k_base",
        "context_filter": "keyword", "context_summarize": cfg.context_summarize,
        "context_token_budget_per_subquery": cfg.context_token_budget,
        "upstream_total_tokens": a, "project_total_tokens": b,
        "total_reduction_pct": 100 * (1 - b / a),
        "median_task_reduction_pct": statistics.median(r["reduction_pct"] for r in rows),
        "task_reduction_min_max_pct": [min(r["reduction_pct"] for r in rows), max(r["reduction_pct"] for r in rows)],
        "upstream_p50_tokens": statistics.median(r["upstream_tokens"] for r in rows),
        "project_p50_tokens": statistics.median(r["project_tokens"] for r in rows),
        "quality_measured": False,
        "limitation": "Token sizes alone do not establish answer quality or total LLM cost; local documentation is not the live research task distribution.",
    }
    return result


async def guard_checks() -> list[dict]:
    from gpt_researcher.actions.query_processing import invoke_retriever_search
    from gpt_researcher.harness.guards import RunGuard, guard_scope

    class RecordedRetriever:
        executions = 0

        def __init__(self, query):
            self.query = query

        def search(self, **kwargs):
            type(self).executions += 1
            time.sleep(0.01)
            return [{"body": "Recorded response", "href": "https://example.org/source"}]

    rows = []
    for duplicates in [0, 20, 60]:
        unique = 200 - duplicates
        calls = [f"recorded query {i}" for i in range(unique)]
        calls += calls[:duplicates]
        random.Random(42).shuffle(calls)
        counts = {}
        for enabled in [False, True]:
            RecordedRetriever.executions = 0
            guard = RunGuard(max_calls=0, max_result_chars=0) if enabled else None
            with guard_scope(guard):
                for query in calls:
                    await invoke_retriever_search(RecordedRetriever(query), 5)
            counts["project" if enabled else "no_guard"] = RecordedRetriever.executions
        rows.append({
            "scenario": "constructed_sequential_trace", "logical_calls": 200,
            "injected_duplicate_calls": duplicates, "external_executions": counts,
            "external_request_reduction_pct": 100 * (1 - counts["project"] / counts["no_guard"]),
        })

    RecordedRetriever.executions = 0
    guard = RunGuard(max_calls=0, max_result_chars=0)
    with guard_scope(guard):
        await asyncio.gather(*[
            invoke_retriever_search(RecordedRetriever("same concurrent query"), 5)
            for _ in range(20)
        ])
    rows.append({
        "scenario": "constructed_20_concurrent_identical_calls",
        "logical_calls": 20, "actual_external_executions": RecordedRetriever.executions,
        "cached_hits": guard.deduped,
        "limitation": "Completed-result cache is not an in-flight request coalescer.",
    })
    return rows


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--offline", action="store_true")
    source.add_argument("--runs", type=Path)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.offline and args.summarize:
        parser.error("--summarize needs real LLM configuration and recorded pages")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for root in [ROOT / "gpt_researcher", ROOT / "multi_agents"]:
        for p in sorted(root.rglob("*.py")):
            hashes[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    dump(out / "manifest.json", {
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "baseline_commit": BASELINE, "current_head": git("rev-parse", "HEAD").strip(),
        "python": sys.version, "mode": "offline" if args.offline else "page_replay",
        "source_sha256": hashes,
    })
    context = await context_replay(args, out)
    guards = await guard_checks()
    summary = {
        "context": context, "guard_mechanism_checks": guards,
        "live_duplicate_call_reduction": {"status": "not_measured", "reason": "Constructed traces do not establish the live duplicate rate."},
        "effective_citations_per_report": {"status": "not_measured", "reason": "Requires generated reports and independent FACT evaluation."},
        "citation_precision": {"status": "not_measured", "reason": "Requires generated reports and independent FACT evaluation."},
    }
    dump(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
