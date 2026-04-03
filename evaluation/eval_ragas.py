from __future__ import annotations
"""
CrimeVision-QA — RAGAS Evaluation

Measures faithfulness, answer relevance, and context precision.

Usage:
    python evaluation/eval_ragas.py \
        --video-id Assault001 \
        --output evaluation/results/ragas_scores.json
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm.agent import run_agent
from llm.retreival_2 import hybrid_search_frames, hybrid_search_transcripts

_QUERIES = [
    "What is happening in the video?",
    "Describe the people involved.",
    "What events occur at the beginning of the video?",
    "Is there any suspicious activity?",
]


async def collect_ragas_data(video_id: str) -> list[dict]:
    """Collect questions, answers, contexts, and ground truths for RAGAS."""
    dataset = []
    for query in _QUERIES:
        result = await run_agent(query, video_id, strategy="zero_shot")

        # Collect context from retrieval
        frames = hybrid_search_frames(query, video_id=video_id, k=5)
        transcripts = hybrid_search_transcripts(query, video_id=video_id, k=3)
        contexts = (
            [d.get("description", "") for d in frames if d.get("description")]
            + [d.get("text", "") for d in transcripts if d.get("text")]
        )

        dataset.append(
            {
                "question": query,
                "answer": result["answer"],
                "contexts": contexts,
                "ground_truth": "",  # Not available — will skip ground_truth metrics
            }
        )
    return dataset


def run_ragas_eval(dataset: list[dict]) -> dict:
    try:
        from datasets import Dataset as HFDataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision

        hf_dataset = HFDataset.from_list(dataset)
        scores = evaluate(
            hf_dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )
        return scores.to_pandas().to_dict(orient="list")

    except ImportError:
        print("[RAGAS] ragas or datasets not installed. Install with: pip install ragas datasets")
        return {"error": "ragas not installed"}
    except Exception as exc:
        print(f"[RAGAS] Evaluation failed: {exc}")
        return {"error": str(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--output", default="evaluation/results/ragas_scores.json")
    args = parser.parse_args()

    print(f"\n[RAGAS] Collecting data for video '{args.video_id}'...")
    dataset = asyncio.run(collect_ragas_data(args.video_id))

    print(f"[RAGAS] Running evaluation on {len(dataset)} queries...")
    scores = run_ragas_eval(dataset)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = {"video_id": args.video_id, "scores": scores, "dataset": dataset}
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[RAGAS] Results saved to {args.output}")
    if "error" not in scores:
        for metric, values in scores.items():
            if isinstance(values, list):
                avg = sum(v for v in values if v is not None) / len(values)
                print(f"  {metric}: {avg:.3f}")


if __name__ == "__main__":
    main()
