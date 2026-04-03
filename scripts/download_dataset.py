from __future__ import annotations
"""
CrimeVision-QA — UCF-Crime Dataset Downloader

Downloads the UCF-Crime dataset from Kaggle using kagglehub and copies
a configurable number of videos per category into ./videos/.

Requirements:
    export KAGGLE_USERNAME=your_username
    export KAGGLE_KEY=your_token
    (or place ~/.kaggle/kaggle.json)

Usage:
    python scripts/download_dataset.py --max-per-category 1 --output-dir ./videos
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# UCF-Crime categories (13 anomaly + 1 normal)
UCF_CATEGORIES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism", "Normal",
]


def download_dataset(output_dir: str = "./videos", max_per_category: int = 1) -> list[dict]:
    """Download UCF-Crime dataset and copy selected videos to *output_dir*.

    Returns a list of dicts with video metadata.
    """
    try:
        import kagglehub
    except ImportError:
        print("kagglehub not installed. Run: pip install kagglehub")
        sys.exit(1)

    # Check credentials
    if not os.getenv("KAGGLE_USERNAME") and not Path("~/.kaggle/kaggle.json").expanduser().exists():
        print(
            "Kaggle credentials not found.\n"
            "Set KAGGLE_USERNAME and KAGGLE_KEY environment variables, or\n"
            "place your kaggle.json at ~/.kaggle/kaggle.json\n"
            "Get token at: https://www.kaggle.com/settings -> 'Create New Token'"
        )
        sys.exit(1)

    print("[Dataset] Downloading UCF-Crime dataset via kagglehub...")
    print("[Dataset] This may take a while (~100GB). Cached after first download.")
    dataset_path = kagglehub.dataset_download("odins0n/ucf-crime-dataset")
    print(f"[Dataset] Dataset path: {dataset_path}")

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Locate the Videos subdirectory
    videos_root = _find_videos_root(dataset_path)
    if not videos_root:
        print(f"[Dataset] Could not locate Videos/ directory under {dataset_path}")
        sys.exit(1)

    print(f"[Dataset] Videos root: {videos_root}")
    copied: list[dict] = []

    for category in UCF_CATEGORIES:
        cat_dir = os.path.join(videos_root, category)
        if not os.path.isdir(cat_dir):
            # Try case-insensitive match
            cat_dir = _find_category_dir(videos_root, category)
        if not cat_dir:
            print(f"[Dataset] Category not found: {category} — skipping")
            continue

        mp4_files = sorted([
            f for f in os.listdir(cat_dir)
            if f.lower().endswith(".mp4")
        ])

        selected = mp4_files[:max_per_category]
        for fname in selected:
            src = os.path.join(cat_dir, fname)
            # Derive clean video_id: strip _x264 suffix
            video_id = fname.replace("_x264.mp4", "").replace(".mp4", "")
            dst = os.path.join(output_dir, fname)

            if os.path.exists(dst):
                print(f"[Dataset] Already exists: {fname}")
            else:
                print(f"[Dataset] Copying {fname} ...")
                shutil.copy2(src, dst)

            copied.append(
                {
                    "video_id": video_id,
                    "filename": fname,
                    "category": category,
                    "path": dst,
                }
            )

    print(f"\n[Dataset] {len(copied)} videos copied to {output_dir}")
    for v in copied:
        print(f"  {v['category']:15s} {v['filename']}")

    return copied


def _find_videos_root(base_path: str) -> str | None:
    """Recursively find the Videos/ directory under base_path."""
    for root, dirs, _ in os.walk(base_path):
        for d in dirs:
            if d.lower() == "videos":
                return os.path.join(root, d)
    return None


def _find_category_dir(videos_root: str, category: str) -> str | None:
    """Case-insensitive directory match."""
    try:
        for entry in os.listdir(videos_root):
            if entry.lower() == category.lower():
                return os.path.join(videos_root, entry)
    except OSError:
        pass
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download UCF-Crime dataset")
    parser.add_argument("--max-per-category", type=int, default=1,
                        help="Max videos to copy per category (default: 1)")
    parser.add_argument("--output-dir", default="./videos",
                        help="Output directory for videos (default: ./videos)")
    args = parser.parse_args()

    download_dataset(args.output_dir, args.max_per_category)


if __name__ == "__main__":
    main()
