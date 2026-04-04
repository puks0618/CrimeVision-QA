from __future__ import annotations
"""
CrimeVision-QA — Kaggle Dataset Ingestion Pipeline

The UCF-Crime dataset on Kaggle contains pre-extracted PNG frames
(not video files). This script:
  1. Finds the dataset in the kagglehub cache
  2. For each category, picks N videos
  3. Subsamples frames (every K-th frame to control API cost)
  4. Copies frames to ./frames/{video_id}/
  5. Runs describe+embed+store via process_frames.py
  6. Updates video_library in MongoDB

Usage:
    python scripts/ingest_from_kaggle.py
    python scripts/ingest_from_kaggle.py --max-per-category 2 --frame-step 20
    python scripts/ingest_from_kaggle.py --max-per-category 1 --frame-step 10 --max-frames 30
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# UCF-Crime categories (kaggle folder names)
UCF_CATEGORIES = [
    ("Abuse",         "Abuse"),
    ("Arrest",        "Arrest"),
    ("Arson",         "Arson"),
    ("Assault",       "Assault"),
    ("Burglary",      "Burglary"),
    ("Explosion",     "Explosion"),
    ("Fighting",      "Fighting"),
    ("NormalVideos",  "Normal"),
    ("RoadAccidents", "RoadAccidents"),
    ("Robbery",       "Robbery"),
    ("Shooting",      "Shooting"),
    ("Shoplifting",   "Shoplifting"),
    ("Stealing",      "Stealing"),
    ("Vandalism",     "Vandalism"),
]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_dataset_root() -> Optional[Path]:
    """Find the kagglehub-cached UCF-Crime dataset."""
    base = Path.home() / ".cache" / "kagglehub" / "datasets" / "odins0n" / "ucf-crime-dataset"
    if not base.exists():
        return None
    # Find latest version
    versions = sorted((base / "versions").iterdir()) if (base / "versions").exists() else []
    if not versions:
        return None
    latest = versions[-1]
    # Check Train or root
    if (latest / "Train").exists():
        return latest / "Train"
    return latest


def get_unique_video_ids(cat_dir: Path) -> list[str]:
    """Extract unique video IDs from PNG filenames in a category directory."""
    video_ids: list[str] = []
    seen: set[str] = set()
    for f in sorted(cat_dir.iterdir()):
        if f.suffix.lower() != ".png":
            continue
        # Name format: Abuse001_x264_900.png
        # video_id = everything before last _<number>
        parts = f.stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            vid_id = parts[0]
            if vid_id not in seen:
                seen.add(vid_id)
                video_ids.append(vid_id)
    return video_ids


def copy_frames(
    cat_dir: Path,
    video_id: str,
    dest_dir: Path,
    frame_step: int = 30,
    max_frames: int = 50,
) -> int:
    """Copy every frame_step-th frame for video_id into dest_dir.

    Returns number of frames copied.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Collect all frames for this video, sorted by frame index
    frames = sorted(
        [f for f in cat_dir.iterdir()
         if f.suffix.lower() == ".png" and f.stem.rsplit("_", 1)[0] == video_id],
        key=lambda f: int(f.stem.rsplit("_", 1)[1]) if f.stem.rsplit("_", 1)[1].isdigit() else 0,
    )

    if not frames:
        print(f"  [ingest] No frames found for {video_id}")
        return 0

    # Subsample
    selected = frames[::frame_step][:max_frames]

    copied = 0
    for src in selected:
        dst = dest_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        copied += 1

    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest UCF-Crime frames into MongoDB")
    parser.add_argument("--max-per-category", type=int, default=1,
                        help="Videos per category (default: 1)")
    parser.add_argument("--frame-step", type=int, default=30,
                        help="Take every Nth frame (default: 30 ≈ 3s at 10fps)")
    parser.add_argument("--max-frames", type=int, default=40,
                        help="Max frames per video (default: 40)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Frame batch size for LLM calls (default: 5)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds between vision API calls (default: 1.0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Copy frames but skip LLM processing")
    args = parser.parse_args()

    # Find dataset
    dataset_root = find_dataset_root()
    if not dataset_root:
        print("[ingest] ERROR: UCF-Crime dataset not found in kagglehub cache.")
        print("  Run: KAGGLE_USERNAME=... KAGGLE_KEY=... python scripts/download_dataset.py")
        sys.exit(1)

    print(f"[ingest] Dataset root: {dataset_root}")
    frames_base = _PROJECT_ROOT / "frames"

    # Import pipeline (after config loaded)
    from llm.process_frames import process_video_frames

    total_processed = 0

    for folder_name, category_label in UCF_CATEGORIES:
        cat_dir = dataset_root / folder_name
        if not cat_dir.exists():
            print(f"[ingest] Category dir not found: {cat_dir} — skipping")
            continue

        video_ids = get_unique_video_ids(cat_dir)[:args.max_per_category]
        if not video_ids:
            print(f"[ingest] No videos found in {folder_name} — skipping")
            continue

        for video_id in video_ids:
            dest_dir = frames_base / video_id
            print(f"\n[ingest] {category_label}/{video_id}")

            # Check if already processed
            from llm.config import frames_col
            existing = frames_col.count_documents({"video_id": video_id})
            if existing > 0:
                print(f"  [ingest] Already ingested ({existing} frames) — skipping")
                total_processed += 1
                continue

            # Copy subsampled frames
            n = copy_frames(cat_dir, video_id, dest_dir,
                           frame_step=args.frame_step,
                           max_frames=args.max_frames)
            print(f"  [ingest] Copied {n} frames to {dest_dir}")

            if n == 0:
                continue

            if args.dry_run:
                print(f"  [ingest] --dry-run: skipping LLM processing")
                continue

            # Describe + embed + store
            result = process_video_frames(
                video_id=video_id,
                frames_dir=str(dest_dir),
                category=category_label,
                batch_size=args.batch_size,
                inter_request_delay=args.delay,
            )
            print(f"  [ingest] Result: {result}")
            total_processed += 1

    print(f"\n[ingest] Done. {total_processed} videos processed.")


if __name__ == "__main__":
    main()
