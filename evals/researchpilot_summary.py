"""Summarize the measured paired campaign; no projected values are inserted."""
from __future__ import annotations

import collections
import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/metrics-2026-11-16-live"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def interval(values):
    values = sorted(v for v in values if v is not None)
    return [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]] if values else None


def summarize():
    tasks = read(OUT / "tasks.json")
    ids = [str(t["id"]) for t in tasks]
    attempts = {}
    for task_id in ids:
        attempts[task_id] = {}
        for arm in ["upstream", "project"]:
            directory = OUT / "reports" / task_id / arm
            status = read(directory / "status.json")
            if status["status"] == "running":
                raise RuntimeError("Report generation is not finished")
            if status["status"] == "completed":
                fact = read(directory / "fact_score.json")
            else:
                fact = {"effective_citations": 0, "eligible_citations": 0, "unknown_citations": 0,
                        "deduplicated_citations": 0}
            events = [json.loads(line) for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            counts = collections.Counter(e["type"] for e in events)
            guards = [e for e in events if e["type"] == "guard" and e.get("allowed")]
            requests = [e for e in events if e["type"] == "provider_request"]
            keys = collections.Counter((e["unit"], json.dumps({**e["options"], "query": " ".join(str(e["options"].get("query", "")).lower().split())}, sort_keys=True)) for e in requests)
            usage = [e["usage"] for e in events if e["type"] == "llm_end" and e.get("usage")]
            attempts[task_id][arm] = {
                "status": status["status"], "seconds": status.get("seconds"),
                "report_words": status.get("report_words", 0),
                "effective_citations": fact["effective_citations"],
                "eligible_citations": fact["eligible_citations"],
                "unknown_citations": fact["unknown_citations"],
                "deduplicated_citations": fact.get("deduplicated_citations", 0),
                "citation_precision": fact.get("citation_precision"),
                "provider_requests": counts["provider_request"], "provider_errors": counts["provider_error"],
                "duplicate_parameter_requests_including_retries": sum(n - 1 for n in keys.values()),
                "cache_hits": sum(e["duplicate"] for e in guards), "guard_allowed_calls": len(guards),
                "research_units": counts["research_start"], "llm_calls": counts["llm_end"],
                "reported_llm_input_tokens": sum(u.get("input_tokens", 0) for u in usage),
                "reported_llm_output_tokens": sum(u.get("output_tokens", 0) for u in usage),
            }

    def metric(sample, arm):
        rows = [attempts[i][arm] for i in sample]
        supported = sum(r["effective_citations"] for r in rows)
        known = sum(r["eligible_citations"] for r in rows)
        all_citations = sum(r["deduplicated_citations"] for r in rows)
        words = sum(r["report_words"] for r in rows)
        return {
            "effective_citations_per_attempt": supported / len(rows),
            "citation_precision_pct": 100 * supported / known if known else None,
            "citation_precision_unknown_as_failure_pct": 100 * supported / all_citations if all_citations else None,
            "effective_citations_per_1000_words": 1000 * supported / words if words else None,
            "provider_requests_per_attempt": sum(r["provider_requests"] for r in rows) / len(rows),
            "mean_report_words": words / len(rows),
        }

    arms = {arm: metric(ids, arm) for arm in ["upstream", "project"]}
    rng = random.Random(42)
    samples = [rng.choices(ids, k=len(ids)) for _ in range(10000)]
    bootstrap = {arm: [metric(sample, arm) for sample in samples] for arm in arms}
    effects = {}
    for key in arms["upstream"]:
        a, b = arms["upstream"][key], arms["project"][key]
        effects[key] = {
            "upstream": a, "project": b, "difference": b - a if a is not None and b is not None else None,
            "upstream_95ci": interval([r[key] for r in bootstrap["upstream"]]),
            "project_95ci": interval([r[key] for r in bootstrap["project"]]),
            "difference_95ci": interval([b_[key] - a_[key] if b_[key] is not None and a_[key] is not None else None for a_, b_ in zip(bootstrap["upstream"], bootstrap["project"])]),
        }

    cache_hits = sum(attempts[i]["project"]["cache_hits"] for i in ids)
    allowed = sum(attempts[i]["project"]["guard_allowed_calls"] for i in ids)
    cache_ci = interval([
        100 * sum(attempts[i]["project"]["cache_hits"] for i in sample) / sum(attempts[i]["project"]["guard_allowed_calls"] for i in sample)
        for sample in samples
    ])

    contexts = {}
    for label, file, variant in [
        ("default_all_baseline_calls", "context_default_all_calls.json", None),
        ("summary_first_two_baseline_calls_per_task", "context_scores.json", "project_summary"),
    ]:
        grouped = collections.defaultdict(lambda: [0, 0, 0])
        rows = read(OUT / file)
        for r in rows:
            a = r["upstream_tokens"] if variant is None else r["tokens"]["upstream_keyword"]
            b = r["project_tokens"] if variant is None else r["tokens"][variant]
            grouped[str(r["task_id"])][0] += a
            grouped[str(r["task_id"])][1] += b
            grouped[str(r["task_id"])][2] += 1
        a = sum(v[0] for v in grouped.values())
        b = sum(v[1] for v in grouped.values())
        ci = interval([100 * (1 - sum(grouped[i][1] for i in sample) / sum(grouped[i][0] for i in sample)) for sample in samples])
        contexts[label] = {"n_calls": len(rows), "n_tasks": len(grouped),
                           "tokenizer": "o200k_base; common reference counter, not DeepSeek billing tokens",
                           "upstream_tokens": a, "project_tokens": b,
                           "reduction_pct": 100 * (1 - b / a), "reduction_95ci_pct": ci,
                           "per_task": dict(grouped)}

    result = {
        "n_paired_tasks": len(ids), "repeats_per_arm_per_task": 1,
        "completed": {arm: sum(attempts[i][arm]["status"] == "completed" for i in ids) for arm in arms},
        "generation_configuration": read(OUT / "manifest.json")["environment_overrides"],
        "method": {
            "baseline_commit": read(OUT / "manifest.json")["baseline_commit"],
            "task_ids": ids,
            "reports": "同模型、同任务、3 个章节，原始开源 LangGraph 与当前项目各运行一次；关闭跨任务记忆和摘要。",
            "citations": "使用官方英文提示词抽取事实-来源对，按 URL 对语义相同事实去重，再对网页正文核验；有效引用不是独立 URL 数。",
            "references": "优先生成时保存的网页正文，缺失时重新抓取；最长正文截取 100000 字符，逐来源记录是否截断。",
            "context": "回放同一批基线网页；默认配置覆盖全部保存的过滤输入，开启摘要的附加实验取每题前两个过滤输入。",
            "cache": "统计当前项目每个 guard 作用域内的可执行逻辑调用和缓存返回；Tavily HTTP 请求另计，包含重试。",
            "statistics": "按任务保持 A/B 配对进行 10000 次 bootstrap，seed=42，取 2.5% 和 97.5% 分位数。",
        },
        "paired_task_bootstrap": {"resamples": 10000, "seed": 42, "ci": "percentile 95%; exploratory task-resampling uncertainty, not a repeated-run interval"},
        "effects": effects, "context_replay": contexts,
        "project_cached_call_displacement": {"cached_calls": cache_hits, "allowed_logical_calls": allowed,
                                              "pct": 100 * cache_hits / allowed, "95ci_pct": cache_ci,
                                              "note": "Same-trace cached-call substitution, not total request reduction versus upstream; retries are separate."},
        "per_task": attempts,
        "limitations": [
            "10 个英文 DRB 任务，每组每题仅运行一次；区间只反映这些任务的重采样波动，不是未来任务表现的窄区间预测。",
            "本次比较 LangGraph 流程；上游公布的 35.2 条/56.0% 使用另一个 deepagents 评测入口和 OpenAI judge，不能直接横比。",
            "保留 FACT 英文提示词；提取使用 DeepSeek chat、核验使用 reasoner，同属一个模型家族，预检时两个别名均返回 deepseek-flash，不能视为独立模型交叉验证。",
            "FACT 提示词将部分支持也计为 supported，精确率排除 unknown；另给出 unknown 计为失败的保守精确率。",
            "优先使用生成时保存的网页正文，缺失时重新抓取公开页面；与原 FACT 抓取后端不同。",
            "报告生成关闭摘要；启用摘要的压缩率来自单独的同网页回放，尚未验证对应报告质量不下降。",
            "上下文 Token 统一按 o200k_base 分词器计数，不是 DeepSeek API 的实际计费 Token。",
            "本轮配置为 keyword/BM25；未配置向量服务，因此没有测试 hybrid 向量召回的收益。",
            "摘要开关启用后仍可能走短输入快速路径或摘要失败回退；这些结果也计入总体压缩率。",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# ResearchPilot 实测结果", "", "10 个固定种子的英文 DeepResearch Bench 任务，每题上游和当前项目各生成一次。", "",
             "## 实际评测方法", ""]
    lines.extend(f"- {key}：{value}" for key, value in result["method"].items())
    lines.extend(["",
             "## 报告与引用", "", "| 指标 | 上游 LangGraph | ResearchPilot | 差值的近似 95% 区间 |", "|---|---:|---:|---:|"]
    )
    for key, title in [
        ("effective_citations_per_attempt", "篇均有效引用"),
        ("citation_precision_pct", "引用精确率（%）"),
        ("citation_precision_unknown_as_failure_pct", "未知来源计失败时精确率（%）"),
        ("effective_citations_per_1000_words", "每千词有效引用"),
        ("provider_requests_per_attempt", "篇均 Tavily 请求数"),
        ("mean_report_words", "篇均词数"),
    ]:
        r = effects[key]
        ci = r["difference_95ci"]
        lines.append(f"| {title} | {r['upstream']:.2f} | {r['project']:.2f} | [{ci[0]:.2f}, {ci[1]:.2f}] |")
    lines += ["", "## 同输入上下文回放", ""]
    for label, row in contexts.items():
        ci = row["reduction_95ci_pct"]
        lines.append(f"- {label}：{row['n_calls']} 次过滤输入，Token {row['upstream_tokens']:,} → {row['project_tokens']:,}，减少 {row['reduction_pct']:.2f}%，任务级重采样区间 [{ci[0]:.2f}%, {ci[1]:.2f}%]。")
    lines += ["", "## 工具调用", "", f"- 当前项目有 {cache_hits}/{allowed} 次可执行逻辑调用由缓存替代，即 {100*cache_hits/allowed:.2f}%。这不是相对上游总请求下降比例。",
              "", "## 口径与限制", ""]
    lines.extend(f"- {text}" for text in result["limitations"])
    lines += ["", "原始数据：每份 report.md、events.jsonl、pages/、fact_score.json；所有 judge 提示词、原始输出与用量在 judge_calls/。", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"effects": effects, "contexts": contexts, "cache": result["project_cached_call_displacement"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    summarize()
