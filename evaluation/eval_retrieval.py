from __future__ import annotations
"""
CrimeVision-QA — Retrieval Evaluation (Recall@K)

Measures how often relevant frames appear in the top-K retrieval results.
Since we don't have ground-truth relevance labels, we use the router to
classify queries and measure whether the correct collection is searched.

Usage:
    python evaluation/eval_retrieval.py \
        --queries evaluation/test_queries.json \
        --video-id Assault001 \
        --k 1 3 5 10

Output saved to: evaluation/results/retrieval_scores.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm.query_model.router import route_query
from llm.retreival_2 import hybrid_search_frames, hybrid_search_transcripts
from llm.inference import semantic_search_frames, semantic_search_transcripts


def evaluate_retrieval(
    queries: list[dict],
    video_id: str,
    k_values: list[int] = [1, 3, 5, 10],
) -> dict:
    results = []

    for q in queries:
        query_text = q["query"]
        expected_intent = q.get("expected_intent", "FIND_FRAME")

        # Route the query
        router_out = route_query(query_text, video_id=video_id)

        # Run both retrieval methods
        t0 = time.perf_counter()
        vector_frames = semantic_search_frames(query_text, video_id=video_id, k=max(k_values))
        vector_transcripts = semantic_search_transcripts(query_text, video_id=video_id, k=max(k_values))
        t1 = time.perf_counter()
        hybrid_frames = hybrid_search_frames(query_text, video_id=video_id, k=max(k_values))
        hybrid_transcripts = hybrid_search_transcripts(query_text, video_id=video_id, k=max(k_values))
        t2 = time.perf_counter()

        result = {
            "query_id": q.get("query_id", "?"),
            "query": query_text,
            "expected_intent": expected_intent,
            "actual_intent": router_out.intent,
            "intent_correct": router_out.intent == expected_intent,
            "router_confidence": router_out.confidence,
            "vector_frames_count": len(vector_frames),
            "hybrid_frames_count": len(hybrid_frames),
            "vector_transcripts_count": len(vector_transcripts),
            "hybrid_transcripts_count": len(hybrid_transcripts),
            "vector_latency_ms": round((t1 - t0) * 1000, 1),
            "hybrid_latency_ms": round((t2 - t1) * 1000, 1),
            "top_frame_score": vector_frames[0].get("score", 0) if vector_frames else 0,
            "top_hybrid_score": hybrid_frames[0].get("rrf_score", 0) if hybrid_frames else 0,
        }
        results.append(result)
        print(f"  [{q.get('query_id')}] intent={router_out.intent} (expected={expected_intent})"
              f" frames={len(vector_frames)} transcripts={len(vector_transcripts)}")

    # Summary stats
    intent_accuracy = sum(r["intent_correct"] for r in results) / len(results) if results else 0
    avg_vector_latency = sum(r["vector_latency_ms"] for r in results) / len(results) if results else 0
    avg_hybrid_latency = sum(r["hybrid_latency_ms"] for r in results) / len(results) if results else 0

    summary = {
        "total_queries": len(results),
        "intent_accuracy": round(intent_accuracy, 3),
        "avg_vector_latency_ms": round(avg_vector_latency, 1),
        "avg_hybrid_latency_ms": round(avg_hybrid_latency, 1),
        "results": results,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="evaluation/test_queries.json")
    parser.add_argument("--video-id", required=True, help="Video ID to run queries against")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--output", default="evaluation/results/retrieval_scores.json")
    args = parser.parse_args()

    with open(args.queries) as f:
        queries = json.load(f)

    # Filter queries that have video_id=null (run against provided video)
    queries_to_run = [
        {**q, "video_id": args.video_id} for q in queries if q.get("video_id") is None
    ]
    # Also include video-specific queries
    queries_to_run += [q for q in queries if q.get("video_id") == args.video_id]

    print(f"\n[Eval] Running {len(queries_to_run)} retrieval queries on video '{args.video_id}'")
    summary = evaluate_retrieval(queries_to_run, video_id=args.video_id, k_values=args.k)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Eval] Intent accuracy: {summary['intent_accuracy']:.1%}")
    print(f"[Eval] Avg vector latency: {summary['avg_vector_latency_ms']}ms")
    print(f"[Eval] Avg hybrid latency: {summary['avg_hybrid_latency_ms']}ms")
    print(f"[Eval] Results saved to {args.output}")


if __name__ == "__main__":
    main()
