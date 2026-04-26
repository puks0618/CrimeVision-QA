from __future__ import annotations
"""
CrimeVision-QA — Frame Processing Orchestrator

Describe → Embed → Store all frames for a video in MongoDB.

Usage:
    python llm/process_frames.py --frames-dir frames/Assault008/ \
                                  --video-id Assault008 \
                                  --category Assault
"""

import argparse
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm.config import frames_col, video_library_col
from llm.gen_frame_desc import describe_frame, _validate_description_quality
from llm.get_voyage_embed import embedding_service

_ZERO_VECTOR = [0.0] * 1024


def _parse_timestamp(frame_file: str) -> float:
    """Extract timestamp in seconds from filename.

    Expected format: 'frame_0015_t30.0s.jpg' -> 30.0
    """
    match = re.search(r"_t([\d.]+)s", frame_file)
    if match:
        return float(match.group(1))
    return 0.0


def _parse_frame_number(frame_file: str) -> int:
    """Extract frame number from filename.

    Expected format: 'frame_0015_t30.0s.jpg' -> 15
    """
    match = re.search(r"frame_(\d+)_", frame_file)
    if match:
        return int(match.group(1))
    return 0


def process_video_frames(
    video_id: str,
    frames_dir: str,
    category: str = "Unknown",
    batch_size: int = 10,
    inter_request_delay: float = 0.5,
) -> dict:
    """Describe, embed, and store all frames in *frames_dir* for *video_id*.

    Returns:
        {"video_id": str, "frames_processed": int, "errors": int}
    """
    frames_dir = os.path.abspath(frames_dir)
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    # Gather all image frame files (JPG or PNG), sorted by frame number
    frame_files = sorted(
        [
            f for f in os.listdir(frames_dir)
            if f.lower().endswith(".jpg") or f.lower().endswith(".png")
        ],
        key=lambda f: _parse_frame_number(f),
    )

    if not frame_files:
        print(f"[Process] No frames (jpg/png) found in {frames_dir}")
        return {"video_id": video_id, "frames_processed": 0, "errors": 0}

    print(f"[Process] Processing {len(frame_files)} frames for video '{video_id}'")

    errors = 0
    processed = 0
    now = datetime.now(timezone.utc)

    # Process in batches: describe batch → embed batch → store batch
    for batch_start in range(0, len(frame_files), batch_size):
        batch_files = frame_files[batch_start : batch_start + batch_size]

        # --- Step 1: Generate descriptions ---
        descriptions: list[str] = []
        meta: list[dict] = []

        for fname in tqdm(batch_files, desc=f"Describing batch {batch_start//batch_size+1}", leave=False):
            path = os.path.join(frames_dir, fname)
            desc = describe_frame(path)
            descriptions.append(desc)
            meta.append(
                {
                    "frame_file": fname,
                    "frame_number": _parse_frame_number(fname),
                    "timestamp_seconds": _parse_timestamp(fname),
                }
            )
            # Small delay between vision API calls to avoid rate limiting
            import time; time.sleep(inter_request_delay)

        # --- Step 2: Embed all descriptions in one batch call ---
        try:
            embeddings = embedding_service.embed(descriptions)
        except Exception as exc:
            print(f"[Process] Embedding batch failed: {exc} — using zero vectors")
            embeddings = [_ZERO_VECTOR] * len(descriptions)
            errors += len(descriptions)

        # --- Step 3: Build and upsert MongoDB documents ---
        from pymongo import UpdateOne
        ops = []
        for m, desc, emb in zip(meta, descriptions, embeddings):
            if desc in ("[DESCRIPTION UNAVAILABLE]", "[INVALID FRAME]"):
                errors += 1
                continue  # Don't store frames with no description
            if emb == _ZERO_VECTOR:
                errors += 1
                continue  # Don't pollute index with zero-vector frames
            
            # Warn about low-quality descriptions but don't skip them
            if not _validate_description_quality(desc):
                print(f"⚠️  Low-quality description for {m['frame_file']}: {desc[:80]}...")

            doc = {
                "video_id": video_id,
                "frame_file": m["frame_file"],
                "frame_number": m["frame_number"],
                "timestamp_seconds": m["timestamp_seconds"],
                "description": desc,
                "embedding": emb,
                "category": category,
                "created_at": now,
            }
            ops.append(
                UpdateOne(
                    {"video_id": video_id, "frame_file": m["frame_file"]},
                    {"$set": doc},
                    upsert=True,
                )
            )

        if ops:
            frames_col.bulk_write(ops, ordered=False)
            processed += len(ops)

    # --- Update video_library metadata ---
    video_library_col.update_one(
        {"video_id": video_id},
        {
            "$set": {
                "video_id": video_id,
                "category": category,
                "frame_count": len(frame_files),
                "processed_at": now,
            }
        },
        upsert=True,
    )

    print(
        f"[Process] Done: {processed} frames stored, {errors} errors "
        f"(video_id='{video_id}')"
    )
    return {"video_id": video_id, "frames_processed": processed, "errors": errors}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Process video frames into MongoDB")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--category", default="Unknown")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between vision API calls (default: 0.5)")
    args = parser.parse_args()

    result = process_video_frames(
        video_id=args.video_id,
        frames_dir=args.frames_dir,
        category=args.category,
        batch_size=args.batch_size,
        inter_request_delay=args.delay,
    )
    print(result)


if __name__ == "__main__":
    main()
