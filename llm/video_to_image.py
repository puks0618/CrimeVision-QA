from __future__ import annotations
"""
CrimeVision-QA — Video Frame Extraction

Extracts frames from a video at configurable intervals using OpenCV.

Usage:
    python llm/video_to_image.py --video videos/Assault008_x264.mp4 \
                                  --output frames/Assault008/ \
                                  --interval 2
"""

import argparse
import os
import sys
from pathlib import Path

import cv2


def extract_frames(
    video_path: str,
    output_dir: str,
    interval_seconds: float = 2.0,
) -> list[dict]:
    """Extract frames from *video_path* every *interval_seconds*.

    Returns a list of dicts:
        [{"frame_file": "frame_0001_t2.0s.jpg",
          "frame_number": 1,
          "timestamp_seconds": 2.0,
          "path": "/absolute/path/to/frame.jpg"}, ...]
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total_frames <= 0:
        print(f"[WARNING] Video has fps={fps}, total_frames={total_frames}. Skipping.")
        cap.release()
        return []

    frame_interval = int(fps * interval_seconds)
    if frame_interval < 1:
        frame_interval = 1

    results: list[dict] = []
    frame_counter = 0
    saved_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_counter % frame_interval == 0:
            saved_counter += 1
            timestamp = frame_counter / fps
            frame_file = f"frame_{saved_counter:04d}_t{timestamp:.1f}s.jpg"
            out_path = os.path.join(output_dir, frame_file)

            cv2.imwrite(out_path, frame)
            results.append(
                {
                    "frame_file": frame_file,
                    "frame_number": saved_counter,
                    "timestamp_seconds": round(timestamp, 1),
                    "path": os.path.abspath(out_path),
                }
            )

        frame_counter += 1

    cap.release()

    duration = total_frames / fps
    print(
        f"[Frames] Extracted {len(results)} frames from {video_path} "
        f"(duration={duration:.1f}s, fps={fps:.1f}, interval={interval_seconds}s)"
    )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from a video file")
    parser.add_argument("--video", required=True, help="Path to the video file")
    parser.add_argument("--output", required=True, help="Output directory for frames")
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds between extracted frames (default: 2)",
    )
    args = parser.parse_args()

    frames = extract_frames(args.video, args.output, args.interval)
    print(f"Total frames saved: {len(frames)}")


if __name__ == "__main__":
    main()
