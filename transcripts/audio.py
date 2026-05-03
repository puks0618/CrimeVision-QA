from __future__ import annotations
"""
CrimeVision-QA — Audio Transcription via Fireworks Whisper-v3

Transcribes an audio file and returns timestamped segments.
Also stores transcript segments in MongoDB when a video_id is supplied.

Usage:
    python transcripts/audio.py --audio transcripts/Assault008.mp3 \
                                  --video-id Assault008
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm.config import (
    EMBED_DIM,
    FIREWORKS_API_KEY,
    FIREWORKS_WHISPER_ENDPOINT,
    FIREWORKS_WHISPER_MODEL,
    transcripts_col,
)
from llm.get_voyage_embed import embedding_service

_MAX_RETRIES = 3
_RETRY_DELAYS = [2, 4, 8]
_MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB — Fireworks limit


def _transcribe_chunk(audio_bytes: bytes, filename: str = "audio.mp3") -> list[dict]:
    """Send audio bytes to Fireworks Whisper and return raw segment dicts."""
    headers = {"Authorization": f"Bearer {FIREWORKS_API_KEY}"}

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.post(
                FIREWORKS_WHISPER_ENDPOINT,
                headers=headers,
                files={"file": (filename, audio_bytes, "audio/mpeg")},
                data={"model": FIREWORKS_WHISPER_MODEL, "response_format": "verbose_json"},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json().get("segments", [])

        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response else "?"
            print(f"[Whisper] HTTP {status} attempt {attempt}/{_MAX_RETRIES}")
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAYS[attempt - 1])

        except requests.exceptions.RequestException as exc:
            print(f"[Whisper] Network error attempt {attempt}/{_MAX_RETRIES}: {exc}")
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAYS[attempt - 1])

    return []


def transcribe_audio(audio_path: str) -> list[dict]:
    """Transcribe *audio_path* and return a list of segment dicts.

    Each segment:
        {"segment_index": int, "start_time": float, "end_time": float, "text": str}

    Returns [] if the file is silent or transcription fails.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    file_size = os.path.getsize(audio_path)
    filename = os.path.basename(audio_path)

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    if file_size > _MAX_AUDIO_BYTES:
        print(f"[Whisper] File {file_size/1e6:.1f}MB > 25MB limit — truncating to 25MB")
        audio_bytes = audio_bytes[:_MAX_AUDIO_BYTES]

    raw_segments = _transcribe_chunk(audio_bytes, filename)

    segments = []
    for i, seg in enumerate(raw_segments):
        text = seg.get("text", "").strip()
        if not text:
            continue
        segments.append(
            {
                "segment_index": i,
                "start_time": round(float(seg.get("start", 0)), 2),
                "end_time": round(float(seg.get("end", 0)), 2),
                "text": text,
            }
        )

    print(f"[Whisper] Transcribed {len(segments)} segments from {audio_path}")
    return segments


def store_transcript_segments(
    video_id: str,
    segments: list[dict],
    batch_size: int = 20,
) -> int:
    """Embed and store transcript segments in MongoDB.

    Returns the number of segments stored.
    """
    if not segments:
        return 0

    texts = [s["text"] for s in segments]

    try:
        embeddings = embedding_service.embed(texts)
    except Exception as exc:
        print(f"[Whisper] Embedding failed: {exc} — storing with zero vectors")
        embeddings = [[0.0] * EMBED_DIM] * len(texts)

    docs = []
    now = datetime.now(timezone.utc)
    for seg, emb in zip(segments, embeddings):
        docs.append(
            {
                "video_id": video_id,
                "segment_index": seg["segment_index"],
                "start_time": seg["start_time"],
                "end_time": seg["end_time"],
                "text": seg["text"],
                "embedding": emb,
                "created_at": now,
            }
        )

    from pymongo import UpdateOne
    stored = 0
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        ops = [
            UpdateOne(
                {"video_id": d["video_id"], "segment_index": d["segment_index"]},
                {"$set": d},
                upsert=True,
            )
            for d in batch
        ]
        transcripts_col.bulk_write(ops, ordered=False)
        stored += len(ops)

    print(f"[Whisper] Stored {stored} transcript segments for video '{video_id}'")
    return stored


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe audio with Whisper-v3")
    parser.add_argument("--audio", required=True, help="Path to the audio file (.mp3)")
    parser.add_argument("--video-id", help="Video ID to store segments in MongoDB")
    args = parser.parse_args()

    segments = transcribe_audio(args.audio)
    for seg in segments[:5]:
        print(f"  [{seg['start_time']:.1f}s-{seg['end_time']:.1f}s] {seg['text'][:80]}")

    if args.video_id:
        store_transcript_segments(args.video_id, segments)


if __name__ == "__main__":
    main()
