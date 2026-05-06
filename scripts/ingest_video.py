from __future__ import annotations
"""
CrimeVision-QA — Single Video Ingestion Pipeline

Given a path to a video file, extracts frames at 2-second intervals,
then runs describe → embed → store for each frame.

Frames are saved to: ./frames/{video_id}/

Usage:
    python scripts/ingest_video.py --video /path/to/video.mp4
    python scripts/ingest_video.py --video /path/to/video.mp4 --video-id MyVideo --category Assault
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from llm.video_to_image import extract_frames
from llm.process_frames import process_video_frames

FRAMES_ROOT = _PROJECT_ROOT / "frames"
FRAME_INTERVAL_SECONDS = 2.0


def ingest_video(
    video_path: str,
    video_id: str | None = None,
    category: str = "Unknown",
    batch_size: int = 10,
    delay: float = 0.5,
) -> dict:
    video_path = str(Path(video_path).resolve())
    video_stem = Path(video_path).stem

    if video_id is None:
        video_id = video_stem

    frames_dir = FRAMES_ROOT / video_id
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Ingest] Video : {video_path}")
    print(f"[Ingest] Video ID : {video_id}")
    print(f"[Ingest] Frames dir: {frames_dir}")
    print(f"[Ingest] Interval  : {FRAME_INTERVAL_SECONDS}s\n")

    # Step 1: extract frames from video
    extracted = extract_frames(video_path, str(frames_dir), FRAME_INTERVAL_SECONDS)
    if not extracted:
        print("[Ingest] No frames extracted. Aborting.")
        return {"video_id": video_id, "frames_processed": 0, "errors": 0}

    print(f"[Ingest] {len(extracted)} frames extracted → {frames_dir}\n")

    # Step 2: describe → embed → store
    result = process_video_frames(
        video_id=video_id,
        frames_dir=str(frames_dir),
        category=category,
        batch_size=batch_size,
        inter_request_delay=delay,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a single video into CrimeVision-QA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="Path to the video file (MP4 etc.)")
    parser.add_argument("--video-id", default=None,
                        help="Unique ID for this video (defaults to filename stem)")
    parser.add_argument("--category", default="Unknown",
                        help="Crime category label (e.g. Assault, Robbery)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Frames per describe+embed batch")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between vision API calls")
    args = parser.parse_args()

    result = ingest_video(
        video_path=args.video,
        video_id=args.video_id,
        category=args.category,
        batch_size=args.batch_size,
        delay=args.delay,
    )
    print(f"\n[Ingest] Result: {result}")


if __name__ == "__main__":
    main()
