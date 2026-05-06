"""
CrimeVision-QA — Query CLI

Ask a natural-language question against a processed video.

Usage:
    python query.py --video-id police_body_cam --query "what is happening in this video?"
    python query.py --video-id police_body_cam --query "describe what the officer is doing" --strategy cot
    python query.py --video-id police_body_cam --query "what was said between 0s and 10s" --strategy few_shot

Strategies:
    zero_shot  — concise 1-2 sentence answer (default)
    cot        — chain-of-thought step-by-step reasoning
    few_shot   — incident-report style with timestamps
    react      — iterative retrieval if first answer is insufficient
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm.agent import run_agent_sync


def main() -> None:
    parser = argparse.ArgumentParser(description="Query a processed video in CrimeVision-QA")
    parser.add_argument("--video-id", required=True, help="Video ID (e.g. police_body_cam)")
    parser.add_argument("--query", required=True, help="Natural-language question")
    parser.add_argument(
        "--strategy",
        default="zero_shot",
        choices=["zero_shot", "cot", "few_shot", "react", "finetuned"],
        help="Reasoning strategy (default: zero_shot)",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Video ID : {args.video_id}")
    print(f"Query    : {args.query}")
    print(f"Strategy : {args.strategy}")
    print(f"{'='*60}\n")

    result = run_agent_sync(
        query=args.query,
        video_id=args.video_id,
        strategy=args.strategy,
    )

    print(f"--- ANSWER ---")
    print(result["answer"])

    if result.get("timestamps"):
        print(f"\n--- TIMESTAMPS ---")
        print(", ".join(f"{t}s" for t in result["timestamps"]))

    if result.get("sources"):
        print(f"\n--- SOURCES ({len(result['sources'])} retrieved) ---")
        for src in result["sources"]:
            if "frame_file" in src:
                print(f"  [Frame] {src.get('frame_file')} @ t={src.get('timestamp_seconds')}s  score={src.get('rrf_score', '?'):.4f}")
            elif "text" in src:
                print(f"  [Audio] {src.get('start_time')}s-{src.get('end_time')}s: \"{src.get('text', '')[:80]}\"")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
