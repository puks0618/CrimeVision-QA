from __future__ import annotations
"""
CrimeVision-QA — Unified Evaluation Runner

Single entry point for all evaluation tasks.

Usage:
    python evaluation/run_eval.py --mode quick    # 3 queries, ~2 min
    python evaluation/run_eval.py --mode full     # all queries, ~15 min
    python evaluation/run_eval.py --mode matrix   # matrix only
    python evaluation/run_eval.py --mode retrieval --video-id police_body_cam
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "evaluation"
RESULTS_DIR = EVAL_DIR / "results"


def run_script(label: str, cmd: list[str]) -> bool:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"\n[run_eval] ❌ {label} failed (exit code {result.returncode})")
        return False
    print(f"\n[run_eval] ✅ {label} complete")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified evaluation runner for CrimeVision-QA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full", "matrix", "retrieval", "latency"],
        default="quick",
        help="Evaluation mode: quick (3 queries), full (all), or individual scripts.",
    )
    parser.add_argument("--video-id", default=None,
                        help="Video ID for retrieval/latency eval (auto-detected if omitted).")
    parser.add_argument("--strategies", default="zero_shot,cot,few_shot,react",
                        help="Comma-separated strategies for matrix eval.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    t0 = time.perf_counter()
    success = True

    if args.mode in ("quick", "full", "matrix"):
        limit_args = ["--limit", "3"] if args.mode == "quick" else []
        ok = run_script(
            f"Evaluation Matrix ({'quick' if args.mode == 'quick' else 'full'})",
            [python, str(EVAL_DIR / "eval_matrix.py"),
             "--strategies", args.strategies] + limit_args,
        )
        success = success and ok

    if args.mode in ("full", "retrieval"):
        video_id = args.video_id
        if not video_id:
            # Auto-detect first available video
            try:
                sys.path.insert(0, str(ROOT))
                from llm.config import video_library_col
                doc = video_library_col.find_one({}, {"video_id": 1})
                video_id = doc["video_id"] if doc else None
            except Exception:
                pass
        if video_id:
            ok = run_script(
                f"Retrieval Evaluation (video={video_id})",
                [python, str(EVAL_DIR / "eval_retrieval.py"),
                 "--video-id", video_id],
            )
            success = success and ok
        else:
            print("[run_eval] ⚠️  No video_id available — skipping retrieval eval")

    if args.mode in ("full", "latency"):
        video_id = args.video_id
        if not video_id:
            try:
                sys.path.insert(0, str(ROOT))
                from llm.config import video_library_col
                doc = video_library_col.find_one({}, {"video_id": 1})
                video_id = doc["video_id"] if doc else None
            except Exception:
                pass
        if video_id:
            ok = run_script(
                f"Latency Benchmark (video={video_id})",
                [python, str(EVAL_DIR / "eval_latency.py"),
                 "--video-id", video_id, "--iterations", "3"],
            )
            success = success and ok
        else:
            print("[run_eval] ⚠️  No video_id available — skipping latency eval")

    elapsed = round(time.perf_counter() - t0, 1)
    print(f"\n{'='*60}")
    print(f"  Evaluation complete in {elapsed}s")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Status: {'✅ All passed' if success else '❌ Some failures'}")
    print(f"{'='*60}\n")

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
