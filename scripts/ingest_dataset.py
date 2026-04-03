from __future__ import annotations
"""
CrimeVision-QA — Dataset Ingestion Pipeline

Reads pre-extracted PNG frames from the kagglehub UCF-Crime cache,
copies a subset to ./frames/{video_id}/, then runs describe → embed → store
for each video using process_frames.py.

The Kaggle dataset (odins0n/ucf-crime-dataset) contains PNG frames already
extracted — no MP4 videos are present, so video_to_image.py is skipped.

Usage:
    python scripts/ingest_dataset.py --max-per-category 2 --max-frames 30
    python scripts/ingest_dataset.py --video-id Abuse001_x264 --category Abuse
    python scripts/ingest_dataset.py --list-videos
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path
from collections import defaultdict

# Project root on sys.path so we can import llm.*
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ---- UCF-Crime categories (13 anomaly + 1 normal) -------------------------
UCF_CATEGORIES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism", "NormalVideos",
]

KAGGLE_CACHE_BASE = Path.home() / ".cache/kagglehub/datasets/odins0n/ucf-crime-dataset/versions/1"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_dataset_root() -> Path:
    """Return the dataset cache root or raise a clear error."""
    if not KAGGLE_CACHE_BASE.exists():
        raise FileNotFoundError(
            f"kagglehub cache not found at {KAGGLE_CACHE_BASE}.\n"
            "Run:  python scripts/download_dataset.py --max-per-category 1"
        )
    return KAGGLE_CACHE_BASE


def _list_category_videos(split: str = "Train") -> dict[str, list[str]]:
    """Return {category: [video_id, ...]} from the kaggle cache."""
    root = _get_dataset_root() / split
    result: dict[str, list[str]] = {}
    for cat in UCF_CATEGORIES:
        cat_dir = root / cat
        if not cat_dir.is_dir():
            continue
        # Get unique video_ids from filenames like Abuse001_x264_0.png
        video_ids: set[str] = set()
        for fname in cat_dir.iterdir():
            if fname.suffix.lower() == ".png":
                # Strip trailing _<framenum>.png to get video_id
                m = re.match(r"^(.+_x264)_\d+\.png$", fname.name)
                if m:
                    video_ids.add(m.group(1))
        result[cat] = sorted(video_ids)
    return result


def _get_frames_for_video(
    video_id: str,
    category: str,
    split: str = "Train",
    max_frames: int | None = None,
    stride: int = 10,
) -> list[Path]:
    """Return sorted list of PNG paths for a single video_id.

    Args:
        stride: sample every Nth frame from the sorted list (default=10).
                The dataset stores every 10 frames (≈1fps), so stride=1 keeps all.
        max_frames: hard cap on frames returned.
    """
    cat_dir = _get_dataset_root() / split / category
    pattern = re.compile(rf"^{re.escape(video_id)}_(\d+)\.png$")
    frames: list[tuple[int, Path]] = []
    for p in cat_dir.iterdir():
        m = pattern.match(p.name)
        if m:
            frames.append((int(m.group(1)), p))
    frames.sort(key=lambda x: x[0])

    # Apply stride to keep every Nth frame
    frames = frames[::stride]

    if max_frames:
        frames = frames[:max_frames]

    return [p for _, p in frames]


def copy_frames_to_local(
    video_id: str,
    category: str,
    frames_out_dir: Path,
    max_frames: int = 30,
    stride: int = 10,
    split: str = "Train",
) -> int:
    """Copy PNG frames from kaggle cache to frames_out_dir. Returns count copied."""
    frames_out_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in frames_out_dir.iterdir() if p.suffix == ".png"}

    src_frames = _get_frames_for_video(video_id, category, split, max_frames, stride)
    if not src_frames:
        print(f"  [ingest] No frames found for {video_id} in {split}/{category}")
        return 0

    copied = 0
    for src in src_frames:
        dst = frames_out_dir / src.name
        if src.name in existing:
            continue  # already there
        shutil.copy2(src, dst)
        copied += 1

    total = len(list(frames_out_dir.glob("*.png")))
    print(f"  [ingest] {video_id}: {copied} new frames copied ({total} total in {frames_out_dir.name}/)")
    return total


# ---------------------------------------------------------------------------
# Main ingestion flow
# ---------------------------------------------------------------------------

def ingest_video(
    video_id: str,
    category: str,
    frames_root: Path,
    max_frames: int = 30,
    stride: int = 10,
    batch_size: int = 5,
    delay: float = 1.0,
    split: str = "Train",
    dry_run: bool = False,
) -> dict:
    """Copy frames then run describe → embed → store for one video."""
    frames_dir = frames_root / video_id
    total_frames = copy_frames_to_local(video_id, category, frames_dir, max_frames, stride, split)

    if total_frames == 0:
        return {"video_id": video_id, "frames_processed": 0, "errors": 0, "skipped": True}

    if dry_run:
        print(f"  [DRY RUN] Would process {total_frames} frames for {video_id}")
        return {"video_id": video_id, "frames_processed": 0, "errors": 0, "dry_run": True}

    # Import here so config loads only when actually ingesting
    from llm.process_frames import process_video_frames

    result = process_video_frames(
        video_id=video_id,
        frames_dir=str(frames_dir),
        category=category,
        batch_size=batch_size,
        inter_request_delay=delay,
    )
    return result


def ingest_all(
    max_per_category: int = 1,
    max_frames: int = 30,
    stride: int = 10,
    batch_size: int = 5,
    delay: float = 1.0,
    split: str = "Train",
    dry_run: bool = False,
    categories: list[str] | None = None,
) -> list[dict]:
    """Ingest all (or selected) categories from the kaggle cache."""
    frames_root = _PROJECT_ROOT / "frames"
    video_map = _list_category_videos(split)

    target_categories = categories or UCF_CATEGORIES
    results: list[dict] = []

    total_videos = sum(
        min(len(vids), max_per_category)
        for cat, vids in video_map.items()
        if cat in target_categories
    )
    print(f"\n[Ingest] Starting ingestion: {total_videos} videos across {len(target_categories)} categories")
    print(f"  Split={split}, max_per_cat={max_per_category}, max_frames={max_frames}, stride={stride}\n")

    for cat in target_categories:
        vids = video_map.get(cat, [])[:max_per_category]
        if not vids:
            print(f"[Ingest] No videos for category: {cat}")
            continue
        for vid in vids:
            print(f"\n[Ingest] Processing: {cat}/{vid}")
            result = ingest_video(
                video_id=vid,
                category=cat,
                frames_root=frames_root,
                max_frames=max_frames,
                stride=stride,
                batch_size=batch_size,
                delay=delay,
                split=split,
                dry_run=dry_run,
            )
            result["category"] = cat
            results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("[Ingest] Summary:")
    total_ok = sum(r.get("frames_processed", 0) for r in results)
    total_err = sum(r.get("errors", 0) for r in results)
    print(f"  Videos processed: {len([r for r in results if not r.get('skipped') and not r.get('dry_run')])}")
    print(f"  Total frames stored: {total_ok}")
    print(f"  Total errors: {total_err}")
    print("=" * 60)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest UCF-Crime dataset frames into MongoDB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--max-per-category", type=int, default=1,
                        help="Max videos to ingest per category")
    parser.add_argument("--max-frames", type=int, default=30,
                        help="Max frames to process per video (after stride)")
    parser.add_argument("--stride", type=int, default=10,
                        help="Sample every Nth frame from sorted frames list")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Frames per describe+embed batch")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds between vision API calls")
    parser.add_argument("--split", default="Train", choices=["Train", "Test"],
                        help="Dataset split to use")
    parser.add_argument("--categories", nargs="+", metavar="CAT",
                        help="Only process specific categories (e.g. Abuse Assault)")
    parser.add_argument("--video-id", help="Process a single specific video by ID")
    parser.add_argument("--category", help="Category for --video-id (required if --video-id set)")
    parser.add_argument("--list-videos", action="store_true",
                        help="List available videos per category and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Copy frames but skip API calls and MongoDB write")
    args = parser.parse_args()

    if args.list_videos:
        video_map = _list_category_videos(args.split)
        print(f"\nAvailable videos in {args.split} split:")
        for cat in UCF_CATEGORIES:
            vids = video_map.get(cat, [])
            print(f"  {cat:15s} ({len(vids):3d} videos): {', '.join(vids[:5])}{'...' if len(vids)>5 else ''}")
        return

    if args.video_id:
        if not args.category:
            parser.error("--category is required when --video-id is set")
        frames_root = _PROJECT_ROOT / "frames"
        result = ingest_video(
            video_id=args.video_id,
            category=args.category,
            frames_root=frames_root,
            max_frames=args.max_frames,
            stride=args.stride,
            batch_size=args.batch_size,
            delay=args.delay,
            split=args.split,
            dry_run=args.dry_run,
        )
        print(result)
        return

    ingest_all(
        max_per_category=args.max_per_category,
        max_frames=args.max_frames,
        stride=args.stride,
        batch_size=args.batch_size,
        delay=args.delay,
        split=args.split,
        dry_run=args.dry_run,
        categories=args.categories,
    )


if __name__ == "__main__":
    main()
