from __future__ import annotations
"""
CrimeVision-QA — Frame Describer Training Data Generator

Pulls existing frame images (from local disk) + their kimi-k2p5 descriptions
(already stored in MongoDB) and saves them as training pairs for Qwen2-VL fine-tuning.

The existing kimi-k2p5 descriptions become the ground-truth labels.
No new API calls are made — this reuses what the ingestion pipeline already produced.

Usage:
    python finetune/generate_frame_data.py \
        --frames-dir ./frames \
        --output finetune/data/frame_training_data.json \
        --min-desc-length 150
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm.config import frames_col, video_library_col

_INSTRUCTION = (
    "Analyze this surveillance frame for criminal activity detection with MAXIMUM detail. "
    "Report: PEOPLE (count, age, skin tone, clothing colors, precise actions, hand positions, "
    "weapons/injuries), VEHICLES (make, model, color, license plate), "
    "OBJECTS (items on ground/in hands, signs of theft, weapons), "
    "SETTING (location type, time of day, weather), "
    "CRITICAL OBSERVATIONS (suspicious behavior, coordination, lookouts, violence indicators). "
    "Be EXTREMELY SPECIFIC and FACTUAL. Write [NOT VISIBLE] for details you cannot see."
)


def generate(frames_dir: str, output_path: str, min_desc_length: int) -> int:
    frames_root = Path(frames_dir)

    # Pull every frame document that has a real description
    cursor = frames_col.find(
        {"description": {"$exists": True, "$ne": "[DESCRIPTION UNAVAILABLE]", "$ne": "[INVALID FRAME]"}},
        {"_id": 0, "video_id": 1, "frame_file": 1, "timestamp_seconds": 1,
         "description": 1, "category": 1},
    )

    examples = []
    skipped_no_image = 0
    skipped_short = 0

    for doc in cursor:
        video_id = doc["video_id"]
        frame_file = doc["frame_file"]
        description = doc.get("description", "")

        # Skip low-quality / too-short descriptions
        if len(description) < min_desc_length:
            skipped_short += 1
            continue

        # Resolve image path: frames/<video_id>/<frame_file>
        image_path = frames_root / video_id / frame_file
        if not image_path.exists():
            skipped_no_image += 1
            continue

        examples.append({
            "image_path": str(image_path),
            "instruction": _INSTRUCTION,
            "description": description,
            "video_id": video_id,
            "timestamp_seconds": doc.get("timestamp_seconds"),
            "category": doc.get("category", "Unknown"),
        })

    print(f"[FrameData] Total examples collected : {len(examples)}")
    print(f"[FrameData] Skipped (image not found): {skipped_no_image}")
    print(f"[FrameData] Skipped (description too short): {skipped_short}")

    if not examples:
        print("[FrameData] No examples found. Make sure:")
        print(f"  1. MongoDB has ingested videos (run test_pipeline.py or POST /api/upload)")
        print(f"  2. Frame images still exist under: {frames_root}/")
        return 0

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    print(f"[FrameData] Saved {len(examples)} training pairs → {output_path}")
    return len(examples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", default="./frames",
                        help="Root directory containing extracted frame images")
    parser.add_argument("--output", default="finetune/data/frame_training_data.json")
    parser.add_argument("--min-desc-length", type=int, default=150,
                        help="Minimum character length of description to include (default: 150)")
    args = parser.parse_args()

    generate(args.frames_dir, args.output, args.min_desc_length)


if __name__ == "__main__":
    main()
