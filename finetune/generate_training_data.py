from __future__ import annotations
"""
CrimeVision-QA — Training Data Generator for QLoRA Fine-Tuning

Generates instruction-response pairs from processed videos for fine-tuning
Llama-3.1-8B-Instruct on law enforcement incident report generation.

Usage:
    python finetune/generate_training_data.py \
        --num-videos 10 \
        --output finetune/data/training_data.json
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm.config import frames_col, video_library_col
from llm.agent import run_agent

_REPORT_INSTRUCTION = (
    "Generate a detailed police-style incident report for the following surveillance "
    "footage description. Include: incident type, timeline of events with timestamps, "
    "subject descriptions (appearance, actions), vehicle descriptions if present, "
    "location details, and key evidence observed. Use formal law enforcement language."
)


async def generate_for_video(video_id: str) -> dict | None:
    """Generate a training example for one video."""
    video_doc = video_library_col.find_one({"video_id": video_id}, {"_id": 0})
    if not video_doc:
        print(f"[Training] Video '{video_id}' not found in library")
        return None

    # Collect frame descriptions as input context
    frames = list(
        frames_col.find(
            {"video_id": video_id},
            {"_id": 0, "frame_file": 1, "timestamp_seconds": 1, "description": 1},
        ).sort("timestamp_seconds", 1).limit(20)
    )

    if not frames:
        print(f"[Training] No frames for video '{video_id}'")
        return None

    # Build input context
    context_lines = [f"Video: {video_id} | Category: {video_doc.get('category', 'Unknown')}", ""]
    for f in frames:
        context_lines.append(
            f"Frame t={f.get('timestamp_seconds', '?')}s: {f.get('description', '')}"
        )
    input_context = "\n".join(context_lines)

    # Generate incident report using the best RAG strategy (CoT)
    result = await run_agent(
        query="Generate a comprehensive incident report for this surveillance footage.",
        video_id=video_id,
        strategy="cot",
    )

    return {
        "instruction": _REPORT_INSTRUCTION,
        "input": input_context,
        "output": result["answer"],
        "video_id": video_id,
        "category": video_doc.get("category", "Unknown"),
    }


async def generate_training_data(num_videos: int, output_path: str) -> int:
    """Generate training examples for up to *num_videos* processed videos."""
    all_videos = list(video_library_col.find({}, {"_id": 0, "video_id": 1}).limit(num_videos))

    if not all_videos:
        print("[Training] No processed videos found. Run the ingestion pipeline first.")
        return 0

    print(f"[Training] Generating training data for {len(all_videos)} videos...")
    examples = []

    for i, v in enumerate(all_videos):
        video_id = v["video_id"]
        print(f"  [{i+1}/{len(all_videos)}] {video_id}")
        example = await generate_for_video(video_id)
        if example:
            examples.append(example)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    print(f"\n[Training] Generated {len(examples)} training examples → {output_path}")
    return len(examples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-videos", type=int, default=10)
    parser.add_argument("--output", default="finetune/data/training_data.json")
    args = parser.parse_args()

    asyncio.run(generate_training_data(args.num_videos, args.output))


if __name__ == "__main__":
    main()
