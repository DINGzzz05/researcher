"""Replay predeclared first two baseline filter calls per DRB task.

All variants see identical saved pages. Runs the real select_context pipeline
and an opt-in summary variant, without changing the user's .env settings.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/metrics-2026-11-16-live"


def save(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")


async def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    import tiktoken
    from gpt_researcher.config import Config
    from gpt_researcher.context.select import select_context
    from evals.researchpilot_metrics import baseline_selector

    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    tasks = json.loads((OUT / "tasks.json").read_text(encoding="utf-8"))
    cfg = Config()
    cfg.context_filter = "keyword"
    cfg.context_dedup = True
    cfg.context_token_budget = 12000
    cfg.context_summarize = False
    encoder = tiktoken.get_encoding("o200k_base")
    upstream = baseline_selector()
    sem = asyncio.Semaphore(2)

    async def evaluate(task_id, index, call):
        target = OUT / "context_replay" / f"{task_id}-{index:04}.json"
        if target.exists():
            return
        async with sem:
            print(json.dumps({"stage": "context_start", "task_id": task_id, "index": index}), flush=True)
            outputs, tokens, costs = {}, {}, {}
            for label, selector, summarize in [
                ("upstream_keyword", upstream, False),
                ("project_default", select_context, False),
                ("project_summary", select_context, True),
            ]:
                config = copy.deepcopy(cfg)
                config.context_summarize = summarize
                spent = []
                outputs[label] = await selector(call["query"], call["pages"], config,
                                                max_results=10, cost_callback=spent.append)
                tokens[label] = len(encoder.encode(outputs[label] or "", disallowed_special=()))
                costs[label] = sum(spent)
            baseline = tokens["upstream_keyword"]
            row = {
                "task_id": task_id, "call_index": index, "query": call["query"],
                "input_sha256": hashlib.sha256(json.dumps(call["pages"], sort_keys=True).encode()).hexdigest(),
                "tokens": tokens, "reported_filter_costs_usd": costs,
                "reduction_pct": {label: 100 * (1 - count / baseline) if baseline else None for label, count in tokens.items()},
                "contexts": outputs,
            }
            save(target, row)
            print(json.dumps({"stage": "context_end", "task_id": task_id, "index": index, "tokens": tokens}), flush=True)

    while True:
        pending = []
        unfinished = 0
        for task in tasks:
            task_id = str(task["id"])
            directory = OUT / "reports" / task_id / "upstream"
            status = directory / "status.json"
            if not status.exists():
                unfinished += 1
                continue
            state = json.loads(status.read_text(encoding="utf-8"))
            if state["status"] == "running":
                unfinished += 1
                continue
            paths = sorted((directory / "pages").glob("*.json"))[:2]
            for path in paths:
                if not (OUT / "context_replay" / f"{task_id}-{int(path.stem):04}.json").exists():
                    pending.append(evaluate(task_id, int(path.stem), json.loads(path.read_text(encoding="utf-8"))))
        if pending:
            await asyncio.gather(*pending)
        if unfinished == 0:
            break
        await asyncio.sleep(10)
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((OUT / "context_replay").glob("*.json"))]
    save(OUT / "context_scores.json", [{k: v for k, v in r.items() if k != "contexts"} for r in rows])
    # Full default-mode replay is cheap and does not need LLM calls. Include
    # all baseline snapshots so the default result is not limited to two calls.
    full_default = []
    for task in tasks:
        task_id = str(task["id"])
        for path in sorted((OUT / "reports" / task_id / "upstream" / "pages").glob("*.json")):
            call = json.loads(path.read_text(encoding="utf-8"))
            config = copy.deepcopy(cfg)
            config.context_summarize = False
            a = await upstream(call["query"], call["pages"], config, max_results=10)
            b = await select_context(call["query"], call["pages"], config, max_results=10)
            full_default.append({"task_id": task_id, "call_index": int(path.stem),
                                 "upstream_tokens": len(encoder.encode(a, disallowed_special=())),
                                 "project_tokens": len(encoder.encode(b, disallowed_special=()))})
    save(OUT / "context_default_all_calls.json", full_default)


if __name__ == "__main__":
    asyncio.run(main())
