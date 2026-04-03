from __future__ import annotations
"""
CrimeVision-QA — Audio Extraction via FFmpeg

Extracts the audio track from a video file as MP3.

Usage:
    python transcripts/video2audio.py --video videos/Assault008_x264.mp4 \
                                       --output transcripts/Assault008.mp3
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _check_ffmpeg() -> None:
    result = subprocess.run(
        ["ffmpeg", "-version"], capture_output=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg is not installed or not on PATH.\n"
            "Install with:  brew install ffmpeg  (macOS)\n"
            "               sudo apt install ffmpeg  (Ubuntu)"
        )


def extract_audio(
    video_path: str,
    output_path: str,
    audio_format: str = "mp3",
) -> str | None:
    """Extract audio from *video_path* and save to *output_path*.

    Returns the output path on success, or None if the video has no audio.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    _check_ffmpeg()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",                   # no video stream
        "-acodec", "libmp3lame",
        "-q:a", "2",             # high quality (0=best, 9=worst)
        "-y",                    # overwrite without asking
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # FFmpeg returns 0 even when there is no audio — check output size
    if not os.path.isfile(output_path) or os.path.getsize(output_path) < 1024:
        print(f"[Audio] No audio track found in {video_path} (or audio is silent)")
        # Clean up empty file
        if os.path.isfile(output_path):
            os.remove(output_path)
        return None

    size_kb = os.path.getsize(output_path) / 1024
    print(f"[Audio] Extracted audio: {output_path} ({size_kb:.1f} KB)")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract audio from a video file")
    parser.add_argument("--video", required=True, help="Path to the video file")
    parser.add_argument("--output", required=True, help="Output audio file path (.mp3)")
    args = parser.parse_args()

    result = extract_audio(args.video, args.output)
    if result is None:
        print("No audio extracted.")
        sys.exit(0)
    print(f"Audio saved to: {result}")


if __name__ == "__main__":
    main()
