"""Live paired LangGraph report runs with isolated upstream/current imports.

Run artifacts contain public research inputs and outputs, never API keys.
Each worker is a separate process so upstream and project packages cannot mix.
"""
from __future__ import annotations

import argparse
import asyncio
import contextvars
import hashlib
import inspect
import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/metrics-2026-11-16-live"
BASELINE = "3582a6841fe7dc801e2a34e34ee408641aea9bce"


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare():
    import requests
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    OUT.mkdir(parents=True, exist_ok=True)
    data_path = OUT / "drb_queries.jsonl"
    if not data_path.exists():
        response = requests.get(
            "https://raw.githubusercontent.com/Ayanami0730/deep_research_bench/main/data/prompt_data/query.jsonl",
            timeout=60,
        )
        response.raise_for_status()
        data_path.write_bytes(response.content)
    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_topic = {}
    for row in rows:
        if row["language"] == "en":
            by_topic.setdefault(row["topic"], []).append(row)
    rng = random.Random(42)
    picked = []
    topics = sorted(by_topic, key=lambda t: -len(by_topic[t]))
    while len(picked) < 10:
        for topic in topics:
            pool = [r for r in by_topic[topic] if r not in picked]
            if pool and len(picked) < 10:
                picked.append(rng.choice(pool))
    save(OUT / "tasks.json", picked)
    config = {
        "FAST_LLM": os.getenv("FAST_LLM", "deepseek:deepseek-chat"),
        "SMART_LLM": os.getenv("SMART_LLM", "deepseek:deepseek-chat"),
        "STRATEGIC_LLM": os.getenv("STRATEGIC_LLM", "deepseek:deepseek-chat"),
        "FAST_TOKEN_LIMIT": "6000", "SMART_TOKEN_LIMIT": "12000", "STRATEGIC_TOKEN_LIMIT": "8000",
        "TEMPERATURE": "0", "MAX_ITERATIONS": "3", "MAX_SEARCH_RESULTS_PER_QUERY": "5",
        "RETRIEVER": "tavily", "CONTEXT_FILTER": "keyword", "CONTEXT_DEDUP": "true",
        "CONTEXT_TOKEN_BUDGET": "12000", "CONTEXT_SUMMARIZE": "false",
        "LONG_TERM_MEMORY": "false", "CITATION_CHECK": "false", "LANGUAGE": "english",
        "TOTAL_WORDS": "1200", "IMAGE_GENERATION_ENABLED": "false", "REPORT_SOURCE": "web",
        "VERBOSE": "false", "LANGCHAIN_TRACING_V2": "false", "MONOCLE_TRACING": "false",
    }
    manifest = {
        "baseline_commit": BASELINE, "sample_size": 10, "seed": 42,
        "dataset_url": "https://raw.githubusercontent.com/Ayanami0730/deep_research_bench/main/data/prompt_data/query.jsonl",
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "environment_overrides": config,
        "task_parameters": {"max_sections": 3, "max_plan_revisions": 3, "max_draft_revisions": 3,
                            "max_fact_check_revisions": 3, "max_claim_searches": 1},
        "source_hashes": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for package in ["gpt_researcher", "multi_agents"] for p in (ROOT / package).rglob("*.py")},
        "note": "This compares upstream/current LangGraph pipelines, not the upstream deepagents example. Generation model is held fixed.",
    }
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ["baseline_commit", "dataset_sha256", "environment_overrides", "task_parameters", "source_hashes"]:
            if previous.get(key) != manifest.get(key):
                raise RuntimeError(f"Cannot mix changed {key} into an existing campaign; use a fresh output directory.")
    else:
        save(manifest_path, manifest)
    return picked, manifest


async def worker(args):
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    os.environ.update(manifest["environment_overrides"])
    # No tracing keys are required by the evaluation; suppress automatic tracing.
    os.environ.pop("LANGCHAIN_API_KEY", None)
    source = Path(args.source_root).resolve()
    sys.path.insert(0, str(source))
    os.chdir(output)

    import tiktoken
    from gpt_researcher import GPTResearcher
    from gpt_researcher.skills.context_manager import ContextManager
    from gpt_researcher.retrievers.tavily.tavily_search import TavilySearch
    from gpt_researcher.llm_provider.generic.base import GenericLLMProvider

    encoder = tiktoken.get_encoding("o200k_base")
    lock = threading.Lock()
    unit = contextvars.ContextVar("metric_research_unit", default="unassigned")
    counter = {"unit": 0, "context": 0, "llm": 0}

    def event(kind, **data):
        row = {"type": kind, "time": time.time(), "unit": unit.get(), **data}
        with lock, (output / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    original_conduct = GPTResearcher.conduct_research

    async def conduct(self, *a, **kw):
        counter["unit"] += 1
        token = unit.set(f"research-{counter['unit']:03}")
        event("research_start", query=self.query)
        try:
            return await original_conduct(self, *a, **kw)
        finally:
            event("research_end", query=self.query)
            unit.reset(token)

    GPTResearcher.conduct_research = conduct
    original_context = ContextManager.get_similar_content_by_query

    async def context(self, query, pages):
        counter["context"] += 1
        index = counter["context"]
        snapshot = {"query": query, "unit": unit.get(), "pages": [
            {"url": p.get("url"), "title": p.get("title"), "raw_content": p.get("raw_content")}
            for p in pages
        ]}
        save(output / "pages" / f"{index:04}.json", snapshot)
        result = await original_context(self, query, pages)
        snapshot["context"] = result
        snapshot["context_tokens"] = len(encoder.encode(result or "", disallowed_special=()))
        save(output / "pages" / f"{index:04}.json", snapshot)
        event("context", index=index, query=query, tokens=snapshot["context_tokens"], pages=len(pages))
        return result

    ContextManager.get_similar_content_by_query = context
    original_search = TavilySearch._search

    def search(self, *a, **kw):
        signature = inspect.signature(original_search).bind_partial(self, *a, **kw)
        signature.apply_defaults()
        values = signature.arguments
        options = {k: values.get(k) for k in ["query", "max_results", "search_depth", "topic", "include_domains", "exclude_domains"]}
        event("provider_request", provider="tavily", options=options)
        try:
            result = original_search(self, *a, **kw)
            event("provider_response", provider="tavily", results=len(result.get("results", [])))
            return result
        except Exception as error:
            event("provider_error", provider="tavily", error_type=type(error).__name__)
            raise

    TavilySearch._search = search
    original_llm = GenericLLMProvider.get_chat_response

    async def llm(self, *a, **kw):
        counter["llm"] += 1
        index = counter["llm"]
        event("llm_start", index=index, model=getattr(self.llm, "model_name", None))
        try:
            text = await original_llm(self, *a, **kw)
            event("llm_end", index=index, usage=getattr(self, "last_usage_metadata", None),
                  metadata=getattr(self, "last_response_metadata", None),
                  output_tokens=len(encoder.encode(text or "", disallowed_special=())))
            return text
        except Exception as error:
            event("llm_error", index=index, error_type=type(error).__name__)
            raise

    GenericLLMProvider.get_chat_response = llm
    if args.arm == "project":
        from gpt_researcher.harness.guards import RunGuard
        original_begin = RunGuard.begin

        def begin(self, key):
            decision = original_begin(self, key)
            event("guard", key=key, allowed=decision.allowed, duplicate=decision.duplicate,
                  reason=decision.reason)
            return decision

        RunGuard.begin = begin

    from multi_agents.agents.orchestrator import ChiefEditorAgent
    rows = json.loads((OUT / "tasks.json").read_text(encoding="utf-8"))
    row = next(r for r in rows if str(r["id"]) == args.task_id)
    model = os.environ["SMART_LLM"].split(":", 1)[1]
    task = {
        "query": row["prompt"], "model": model, "max_sections": 3,
        "max_plan_revisions": 3, "max_draft_revisions": 3, "max_fact_check_revisions": 3,
        "max_claim_searches": 1, "publish_formats": {"markdown": True, "pdf": False, "docx": False},
        "include_human_feedback": False, "follow_guidelines": False, "guidelines": [],
        "verbose": False, "source": "web",
    }
    save(output / "task.json", task)
    started = time.perf_counter()
    result = {"arm": args.arm, "task_id": args.task_id, "topic": row["topic"], "status": "running"}
    save(output / "status.json", result)
    try:
        agent = ChiefEditorAgent(task)
        state = await asyncio.wait_for(agent.run_research_task(), timeout=1800)
        report = state.get("report", "")
        (output / "report.md").write_text(report, encoding="utf-8")
        result.update(status="completed" if report.strip() else "empty_report",
                      report_words=len(report.split()), report_tokens=len(encoder.encode(report, disallowed_special=())))
        save(output / "state.json", state)
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__)
        # Traceback records call sites without echoing request credentials.
        import traceback
        frames = traceback.extract_tb(error.__traceback__)
        save(output / "failure_frames.json", [{"file": f.filename, "line": f.lineno, "function": f.name} for f in frames])
        event("task_error", error_type=type(error).__name__)
    finally:
        result["seconds"] = round(time.perf_counter() - started, 2)
        save(output / "status.json", result)
    print(json.dumps(result, ensure_ascii=True), flush=True)


async def campaign(args):
    rows, manifest = prepare()
    sem = asyncio.Semaphore(args.concurrency)
    jobs = []
    rng = random.Random(42)
    for row in rows[:args.limit]:
        arms = ["upstream", "project"]
        rng.shuffle(arms)
        for arm in arms:
            jobs.append((row, arm))

    async def one(row, arm):
        path = OUT / "reports" / str(row["id"]) / arm
        status = path / "status.json"
        if status.exists() and json.loads(status.read_text(encoding="utf-8")).get("status") == "completed":
            return
        async with sem:
            path.mkdir(parents=True, exist_ok=True)
            source = ROOT if arm == "project" else OUT / "upstream"
            print(json.dumps({"stage": "start", "arm": arm, "id": row["id"]}), flush=True)
            with (path / "console.log").open("wb") as log:
                process = await asyncio.create_subprocess_exec(
                    sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "worker",
                    "--source-root", str(source), "--output", str(path), "--arm", arm,
                    "--task-id", str(row["id"]), stdout=log, stderr=asyncio.subprocess.STDOUT,
                )
                await process.wait()
            print(json.dumps({"stage": "end", "arm": arm, "id": row["id"], "exit_code": process.returncode,
                              "status": json.loads(status.read_text(encoding="utf-8")) if status.exists() else "no_status"}), flush=True)

    await asyncio.gather(*(one(row, arm) for row, arm in jobs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("campaign")
    c.add_argument("--limit", type=int, default=10)
    c.add_argument("--concurrency", type=int, default=2)
    w = sub.add_parser("worker")
    w.add_argument("--source-root", required=True)
    w.add_argument("--output", required=True)
    w.add_argument("--arm", choices=["upstream", "project"], required=True)
    w.add_argument("--task-id", required=True)
    args = parser.parse_args()
    asyncio.run(campaign(args) if args.mode == "campaign" else worker(args))


if __name__ == "__main__":
    main()
