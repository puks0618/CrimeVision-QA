from __future__ import annotations
"""
CrimeVision-QA — Whisper Fine-Tuning Data Generator

Extracts individual audio segments from local MP3 files and pairs them
with the ground-truth transcriptions already stored in MongoDB.

Each (audio_clip, transcript_text) pair becomes one Whisper training example.
No new API calls — reuses what the ingestion pipeline already produced.

Usage:
    python finetune/generate_whisper_data.py \
        --frames-dir ./frames \
        --output finetune/data/whisper_training_data.json

Requirements:
    pip install ffmpeg-python
    Install ffmpeg and add it to your system's PATH
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm.config import transcripts_col, video_library_col


def _resolve_ffmpeg() -> str | None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    local_appdata = os.getenv("LOCALAPPDATA")
    if local_appdata:
        winget_root = Path(local_appdata) / "Microsoft" / "WinGet" / "Packages"
        matches = sorted(winget_root.glob("**/ffmpeg.exe"))
        if matches:
            return str(matches[0])
    return None


def _ffmpeg_available() -> bool:
    ffmpeg_path = _resolve_ffmpeg()
    if not ffmpeg_path:
        return False
    try:
        result = subprocess.run([ffmpeg_path, "-version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def _find_audio_file(frames_root: Path, video_id: str) -> Path | None:
    video_dir = frames_root / video_id
    candidates = [
        video_dir / f"{video_id}.mp3",
        video_dir / "audio.mp3",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(video_dir.glob("*.mp3"))
    return matches[0] if matches else None


def _extract_segment(ffmpeg_path: str, audio_path: str, start: float, end: float, out_path: str) -> bool:
    """Cut [start, end] seconds from audio_path into out_path (wav for Whisper)."""
    duration = end - start
    if duration <= 0.1:
        return False
    cmd = [
        ffmpeg_path, "-y",
        "-ss", str(start),
        "-t", str(duration),
        "-i", audio_path,
        "-ar", "16000",   # Whisper expects 16kHz
        "-ac", "1",       # mono
        "-c:a", "pcm_s16le",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    return result.returncode == 0 and os.path.getsize(out_path) > 512


def generate(frames_dir: str, output_path: str, min_text_length: int) -> int:
    if not _ffmpeg_available():
        print("[WhisperData] ffmpeg not found on PATH. Install ffmpeg and reopen the shell.")
        sys.exit(1)
    ffmpeg_path = _resolve_ffmpeg()
    if ffmpeg_path is None:
        print("[WhisperData] ffmpeg could not be resolved.")
        sys.exit(1)

    frames_root = Path(frames_dir)

    # Pull all transcript segments from MongoDB
    all_segments = list(
        transcripts_col.find(
            {"text": {"$exists": True}},
            {"_id": 0, "video_id": 1, "segment_index": 1,
             "start_time": 1, "end_time": 1, "text": 1},
        )
    )

    if not all_segments:
        print("[WhisperData] No transcript segments found in MongoDB.")
        print("  Run the ingestion pipeline first (POST /api/upload or test_pipeline.py).")
        return 0

    print(f"[WhisperData] Found {len(all_segments)} transcript segments in MongoDB")

    # Group by video_id to avoid re-opening the same audio file
    by_video: dict[str, list[dict]] = {}
    for seg in all_segments:
        by_video.setdefault(seg["video_id"], []).append(seg)

    examples = []
    clip_dir = Path(output_path).parent / "whisper_clips"
    clip_dir.mkdir(parents=True, exist_ok=True)

    for video_id, segments in by_video.items():
        audio_path = _find_audio_file(frames_root, video_id)
        if audio_path is None:
            print(f"  [WhisperData] Audio not found for {video_id} under {frames_root / video_id}")
            continue

        print(f"  [WhisperData] Processing {video_id} ({len(segments)} segments)...")
        for seg in segments:
            text = seg.get("text", "").strip()
            if len(text) < min_text_length:
                continue

            start = float(seg.get("start_time", 0))
            end = float(seg.get("end_time", 0))
            clip_name = f"{video_id}_seg{seg['segment_index']:04d}.wav"
            clip_path = str(clip_dir / clip_name)

            if _extract_segment(ffmpeg_path, str(audio_path), start, end, clip_path):
                examples.append({
                    "audio_path": clip_path,
                    "text": text,
                    "video_id": video_id,
                    "start_time": start,
                    "end_time": end,
                    "segment_index": seg["segment_index"],
                })

    print(f"\n[WhisperData] Extracted {len(examples)} audio clips → {clip_dir}/")

    if not examples:
        print("[WhisperData] No examples generated. Check audio files exist under frames/")
        return 0

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    print(f"[WhisperData] Saved {len(examples)} training pairs → {output_path}")
    return len(examples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", default="./frames")
    parser.add_argument("--output", default="finetune/data/whisper_training_data.json")
    parser.add_argument("--min-text-length", type=int, default=10,
                        help="Minimum transcript text length to include (default: 10 chars)")
    args = parser.parse_args()

    generate(args.frames_dir, args.output, args.min_text_length)


if __name__ == "__main__":
    main()
