from __future__ import annotations
"""
CrimeVision-QA — Prompting Strategy Comparison

Runs the same test queries through all 4 prompting strategies and
compares output quality.

Usage:
    python evaluation/eval_prompts.py \
        --video-id Assault001 \
        --strategies zero_shot cot few_shot react \
        --output evaluation/results/prompt_comparison.json
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm.agent import run_agent

_TEST_QUERIES = [
    "What is happening in this video?",
    "Describe the sequence of events with timestamps.",
    "What does the person involved look like?",
    "Is there any audio evidence? What was said?",
]


async def _run_single(query: str, video_id: str, strategy: str) -> dict:
    t0 = time.perf_counter()
    result = await run_agent(query, video_id, strategy)
    elapsed = round(time.perf_counter() - t0, 2)
    return {
        "strategy": strategy,
        "query": query,
        "answer": result["answer"],
        "timestamps_found": len(result.get("timestamps", [])),
        "sources_count": len(result.get("sources", [])),
        "latency_s": elapsed,
    }


async def evaluate_strategies(
    video_id: str,
    strategies: list[str],
    queries: list[str],
) -> list[dict]:
    results = []
    for query in queries:
        print(f"\n  Query: {query[:60]}...")
        for strategy in strategies:
            print(f"    [{strategy}] ", end="", flush=True)
            r = await _run_single(query, video_id, strategy)
            results.append(r)
            print(f"{r['latency_s']}s — {r['answer'][:80]}...")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["zero_shot", "cot", "few_shot", "react"],
    )
    parser.add_argument("--output", default="evaluation/results/prompt_comparison.json")
    args = parser.parse_args()

    print(f"\n[Eval] Comparing {len(args.strategies)} strategies on {len(_TEST_QUERIES)} queries")
    results = asyncio.run(
        evaluate_strategies(args.video_id, args.strategies, _TEST_QUERIES)
    )

    # Aggregate by strategy
    summary: dict = {}
    for r in results:
        s = r["strategy"]
        if s not in summary:
            summary[s] = {"latencies": [], "timestamps_found": [], "answers": []}
        summary[s]["latencies"].append(r["latency_s"])
        summary[s]["timestamps_found"].append(r["timestamps_found"])
        summary[s]["answers"].append(r["answer"])

    strategy_summary = {
        s: {
            "avg_latency_s": round(sum(v["latencies"]) / len(v["latencies"]), 2),
            "avg_timestamps": round(sum(v["timestamps_found"]) / len(v["timestamps_found"]), 1),
        }
        for s, v in summary.items()
    }

    output = {"strategy_summary": strategy_summary, "detailed_results": results}

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print("\n[Eval] Strategy Summary:")
    for s, m in strategy_summary.items():
        print(f"  {s:15s}: avg_latency={m['avg_latency_s']}s  avg_timestamps={m['avg_timestamps']}")
    print(f"\n[Eval] Full results saved to {args.output}")


if __name__ == "__main__":
    main()
