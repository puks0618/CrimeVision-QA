from __future__ import annotations
"""
CrimeVision-QA — Latency Benchmarks

Measures per-stage and end-to-end latency.

Usage:
    python evaluation/eval_latency.py \
        --video-id Assault001 \
        --iterations 5 \
        --output evaluation/results/latency.json
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm.get_voyage_embed import embedding_service
from llm.query_model.router import route_query
from llm.retreival_2 import hybrid_search_frames
from llm.agent import run_agent

_TEST_QUERY = "What is happening in this video?"


def benchmark_stage(name: str, fn, iterations: int) -> dict:
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "stage": name,
        "iterations": iterations,
        "avg_ms": round(sum(latencies) / len(latencies), 1),
        "min_ms": round(min(latencies), 1),
        "max_ms": round(max(latencies), 1),
        "p95_ms": round(sorted(latencies)[int(0.95 * len(latencies))], 1),
    }


async def benchmark_e2e(video_id: str, strategy: str, iterations: int) -> dict:
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        await run_agent(_TEST_QUERY, video_id, strategy)
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "stage": f"e2e_{strategy}",
        "iterations": iterations,
        "avg_ms": round(sum(latencies) / len(latencies), 1),
        "min_ms": round(min(latencies), 1),
        "max_ms": round(max(latencies), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", default="evaluation/results/latency.json")
    args = parser.parse_args()

    print(f"\n[Latency] Benchmarking {args.iterations} iterations each...\n")
    results = []

    # Embedding latency
    print("  Stage: embedding...")
    r = benchmark_stage(
        "embedding",
        lambda: embedding_service.embed([_TEST_QUERY]),
        args.iterations,
    )
    results.append(r)
    print(f"    avg={r['avg_ms']}ms  p95={r['p95_ms']}ms")

    # Routing latency
    print("  Stage: routing...")
    r = benchmark_stage(
        "routing",
        lambda: route_query(_TEST_QUERY),
        args.iterations,
    )
    results.append(r)
    print(f"    avg={r['avg_ms']}ms  p95={r['p95_ms']}ms")

    # Hybrid retrieval latency
    print("  Stage: hybrid retrieval...")
    r = benchmark_stage(
        "hybrid_retrieval",
        lambda: hybrid_search_frames(_TEST_QUERY, video_id=args.video_id, k=5),
        args.iterations,
    )
    results.append(r)
    print(f"    avg={r['avg_ms']}ms  p95={r['p95_ms']}ms")

    # End-to-end for each strategy
    for strategy in ["zero_shot", "cot"]:
        print(f"  Stage: e2e ({strategy})...")
        r = asyncio.run(benchmark_e2e(args.video_id, strategy, args.iterations))
        results.append(r)
        print(f"    avg={r['avg_ms']}ms")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"video_id": args.video_id, "benchmarks": results}, f, indent=2)

    print(f"\n[Latency] Results saved to {args.output}")


if __name__ == "__main__":
    main()
