"""
Standalone test pipeline — video → frames + audio → describe → embed → MongoDB.
No config.py imports. Reads all credentials directly from .env.

Usage:
    python test_pipeline.py --video path/to/video.mp4
    python test_pipeline.py --video path/to/video.mp4 --interval 2 --max-frames 5 --category Assault
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pymongo
from dotenv import load_dotenv
from pymongo import UpdateOne

load_dotenv(Path(__file__).parent / ".env")

FIREWORKS_API_KEY  = os.environ["FIREWORKS_API_KEY"]
VOYAGE_API_KEY     = os.environ["VOYAGE_API_KEY"]
MONGODB_URI        = os.environ["MONGODB_URI"]
MONGODB_DB_NAME    = os.getenv("MONGODB_DB_NAME", "video_intelligence")

FIREWORKS_API_BASE        = "https://api.fireworks.ai/inference/v1"
FIREWORKS_VISION_MODEL    = "accounts/fireworks/models/kimi-k2p5"
FIREWORKS_WHISPER_ENDPOINT = "https://audio-prod.api.fireworks.ai/v1/audio/transcriptions"
FIREWORKS_WHISPER_MODEL   = "whisper-v3"
VOYAGE_EMBED_MODEL        = "voyage-3-large"

_PROMPT = (
    "Analyze this surveillance frame for criminal activity detection with MAXIMUM detail. "
    "MANDATORY - Report with precision:\n\n"
    "PEOPLE (if visible):\n"
    "- Count, approximate age range, skin tone, ethnicity\n"
    "- Exact clothing: colors, types, visible logos/text, accessories\n"
    "- PRECISE ACTIONS: running/walking/standing/fighting/stealing/concealing/pointing/attacking/fleeing\n"
    "- Hand positions: empty, carrying items, using weapons, raised, in pockets\n"
    "- Visible face details: mask/hoodie obscuring face, visible features, jewelry\n"
    "- Any visible weapons, tools, or stolen items\n"
    "- Injuries, blood, or suspicious marks\n"
    "- Direction of movement with compass direction if possible\n\n"
    "VEHICLES (if visible):\n"
    "- Make, model, color, condition\n"
    "- License plate: any visible digits/characters\n"
    "- Windows tinted/normal, occupants visible\n"
    "- Damage, modifications, unique features\n"
    "- Direction/position relative to scene\n\n"
    "OBJECTS/PROPERTY:\n"
    "- Items on ground, in hands, or being transported\n"
    "- Signs of theft: open doors, broken windows, scattered items\n"
    "- Weapons visible: guns, knives, bats, explosives\n"
    "- Packages, boxes, bags - color and contents if identifiable\n\n"
    "SETTING:\n"
    "- Location type: street/parking lot/store/home/warehouse/alley\n"
    "- Entry/exit points visible\n"
    "- Time of day indicators: sunlight/darkness/street lights\n"
    "- Weather: dry/wet/snowing\n"
    "- Surrounding buildings, signs, landmarks\n\n"
    "CRITICAL OBSERVATIONS:\n"
    "- Is there suspicious behavior? Be specific.\n"
    "- Are multiple people coordinating movements?\n"
    "- Anyone acting as lookout or sentinel?\n"
    "- Signs of organized crime vs opportunistic theft?\n"
    "- Any violence, weapons, or imminent danger indicators?\n\n"
    "IMPORTANT: Be EXTREMELY SPECIFIC and FACTUAL. If you cannot clearly see a detail, "
    "write '[NOT VISIBLE]' instead of guessing."
)


# ---------------------------------------------------------------------------
# Step 1a: Extract frames from video
# ---------------------------------------------------------------------------

def extract_frames(video_path: str, output_dir: str, interval_seconds: float, max_frames: int) -> list[dict]:
    import cv2

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    print(f"  FPS: {fps:.1f}  |  Total frames: {total_frames}  |  Duration: {duration:.1f}s")

    frame_interval = max(1, int(fps * interval_seconds))
    results = []
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
            results.append({"frame_file": frame_file, "timestamp": round(timestamp, 1), "path": out_path})
            if max_frames and saved_counter >= max_frames:
                break
        frame_counter += 1

    cap.release()
    return results


# ---------------------------------------------------------------------------
# Step 1b: Extract audio + transcribe
# ---------------------------------------------------------------------------

def extract_audio(video_path: str, audio_path: str) -> str | None:
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True)
    if result.returncode != 0:
        print("  [Audio] ffmpeg not found. Install with: brew install ffmpeg")
        return None

    os.makedirs(os.path.dirname(os.path.abspath(audio_path)), exist_ok=True)
    cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "libmp3lame", "-q:a", "2", "-y", audio_path]
    subprocess.run(cmd, capture_output=True)

    if not os.path.isfile(audio_path) or os.path.getsize(audio_path) < 1024:
        print("  [Audio] No audio track found in video.")
        if os.path.isfile(audio_path):
            os.remove(audio_path)
        return None

    print(f"  [Audio] Extracted: {audio_path} ({os.path.getsize(audio_path)/1024:.1f} KB)")
    return audio_path


def transcribe_audio(audio_path: str) -> list[dict]:
    import requests

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    if len(audio_bytes) > 25 * 1024 * 1024:
        audio_bytes = audio_bytes[:25 * 1024 * 1024]

    headers = {"Authorization": f"Bearer {FIREWORKS_API_KEY}"}

    for attempt in range(1, 4):
        try:
            resp = requests.post(
                FIREWORKS_WHISPER_ENDPOINT,
                headers=headers,
                files={"file": (os.path.basename(audio_path), audio_bytes, "audio/mpeg")},
                data={"model": FIREWORKS_WHISPER_MODEL, "response_format": "verbose_json"},
                timeout=120,
            )
            resp.raise_for_status()
            raw_segments = resp.json().get("segments", [])
            segments = []
            for i, seg in enumerate(raw_segments):
                text = seg.get("text", "").strip()
                if text:
                    segments.append({
                        "segment_index": i,
                        "start_time": round(float(seg.get("start", 0)), 2),
                        "end_time": round(float(seg.get("end", 0)), 2),
                        "text": text,
                    })
            print(f"  [Whisper] Transcribed {len(segments)} segments")
            return segments
        except Exception as exc:
            print(f"  [Whisper] Attempt {attempt}/3 failed: {exc}")
            if attempt < 3:
                time.sleep(2 ** attempt)

    return []


def segments_at_timestamp(segments: list[dict], timestamp: float, window: float = 2.0) -> list[dict]:
    t_end = timestamp + window
    return [s for s in segments if s["start_time"] < t_end and s["end_time"] > timestamp]


# ---------------------------------------------------------------------------
# Step 2: Describe a frame
# ---------------------------------------------------------------------------

def describe_frame(frame_path: str) -> str:
    import requests

    ext = Path(frame_path).suffix.lower()
    mime_type = "image/png" if ext == ".png" else "image/jpeg"

    with open(frame_path, "rb") as f:
        b64_data = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": FIREWORKS_VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}},
                ],
            }
        ],
        "max_tokens": 1000,
        "temperature": 0.1,
    }

    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(1, 4):
        try:
            resp = requests.post(
                f"{FIREWORKS_API_BASE}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            # kimi-k2p5 outputs reasoning inline — keep only from first section header
            match = re.search(r"(PEOPLE[:\s]|VEHICLES[:\s]|SETTING[:\s]|OBJECTS[:\s])", raw)
            if match:
                raw = raw[match.start():].strip()
            return raw
        except Exception as exc:
            print(f"    [Vision] Attempt {attempt}/3 failed: {exc}")
            if attempt < 3:
                time.sleep(2 ** attempt)

    return "[DESCRIPTION UNAVAILABLE]"


# ---------------------------------------------------------------------------
# Step 3: Embed a frame description
# ---------------------------------------------------------------------------

def embed_text(text: str) -> list[float]:
    import voyageai
    client = voyageai.Client(api_key=VOYAGE_API_KEY)
    for attempt in range(1, 4):
        try:
            result = client.embed([text], model=VOYAGE_EMBED_MODEL, input_type="document")
            return result.embeddings[0]
        except Exception as exc:
            err = str(exc).lower()
            if "rate" in err or "429" in err or "limit" in err:
                wait = attempt * 25
                print(f"    [Voyage] Rate limited, waiting {wait}s (attempt {attempt}/3)...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Voyage embedding failed after 3 retries")


# ---------------------------------------------------------------------------
# Step 4: Store everything to MongoDB
# ---------------------------------------------------------------------------

def store_to_mongodb(
    video_id: str,
    category: str,
    frame_results: list[dict],
    transcript_segments: list[dict],
) -> None:
    import voyageai

    client = pymongo.MongoClient(MONGODB_URI)
    try:
        db              = client[MONGODB_DB_NAME]
        frames_col      = db["video_intelligence"]
        transcripts_col = db["video_intelligence_transcripts"]
        library_col     = db["video_library"]
        now             = datetime.now(timezone.utc)

        # 4a — Embed all transcript segments in one Voyage batch call
        if transcript_segments:
            print(f"  [Voyage] Embedding {len(transcript_segments)} transcript segments...")
            texts = [seg["text"] for seg in transcript_segments]
            voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
            segment_embeddings = None

            for attempt in range(1, 4):
                try:
                    result = voyage_client.embed(texts, model=VOYAGE_EMBED_MODEL, input_type="document")
                    segment_embeddings = result.embeddings
                    print(f"  [Voyage] Transcript embeddings done ({len(segment_embeddings)} vectors)")
                    break
                except Exception as exc:
                    err = str(exc).lower()
                    if "rate" in err or "429" in err or "limit" in err:
                        wait = attempt * 25
                        print(f"  [Voyage] Rate limited on transcripts, waiting {wait}s (attempt {attempt}/3)...")
                        time.sleep(wait)
                    else:
                        raise

            # Stitch embeddings back onto segments in place (also updates results.json)
            if segment_embeddings:
                for seg, emb in zip(transcript_segments, segment_embeddings):
                    seg["embedding"] = emb

        # 4b — Upsert frames → video_intelligence
        frame_ops = []
        for fr in frame_results:
            if fr["embedding"] is None:
                continue
            m = re.search(r"frame_(\d+)_", fr["frame_file"])
            frame_number = int(m.group(1)) if m else None
            frame_ops.append(UpdateOne(
                {"video_id": video_id, "frame_file": fr["frame_file"]},
                {"$set": {
                    "video_id":          video_id,
                    "frame_file":        fr["frame_file"],
                    "frame_number":      frame_number,
                    "timestamp_seconds": fr["timestamp_seconds"],
                    "description":       fr["description"],
                    "embedding":         fr["embedding"],
                    "category":          category,
                    "created_at":        now,
                }},
                upsert=True,
            ))

        if frame_ops:
            res = frames_col.bulk_write(frame_ops, ordered=False)
            print(f"  [MongoDB] video_intelligence: {res.upserted_count} inserted, {res.modified_count} updated")

        # 4c — Upsert transcript segments → video_intelligence_transcripts
        transcript_ops = []
        for seg in transcript_segments:
            transcript_ops.append(UpdateOne(
                {"video_id": video_id, "segment_index": seg["segment_index"]},
                {"$set": {
                    "video_id":      video_id,
                    "segment_index": seg["segment_index"],
                    "start_time":    seg["start_time"],
                    "end_time":      seg["end_time"],
                    "text":          seg["text"],
                    "embedding":     seg.get("embedding"),
                    "created_at":    now,
                }},
                upsert=True,
            ))

        if transcript_ops:
            res = transcripts_col.bulk_write(transcript_ops, ordered=False)
            print(f"  [MongoDB] video_intelligence_transcripts: {res.upserted_count} inserted, {res.modified_count} updated")

        # 4d — Upsert video metadata → video_library
        library_col.update_one(
            {"video_id": video_id},
            {"$set": {
                "video_id":     video_id,
                "category":     category,
                "frame_count":  len(frame_results),
                "processed_at": now,
            }},
            upsert=True,
        )
        print(f"  [MongoDB] video_library: upserted video_id={video_id!r}")

    finally:
        client.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Full pipeline: video → frames+audio → describe → embed → MongoDB")
    parser.add_argument("--video", required=True, help="Path to video file (mp4 etc.)")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between frames (default: 2)")
    parser.add_argument("--max-frames", type=int, default=3, help="Max frames to process (default: 3)")
    parser.add_argument("--category", default="Unknown", help="Crime category label (default: Unknown)")
    args = parser.parse_args()

    video_path = args.video
    if not Path(video_path).is_file():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    video_id   = Path(video_path).stem
    frames_dir = str(Path("frames") / video_id)
    audio_path = str(Path(frames_dir) / f"{video_id}.mp3")

    print(f"\n{'='*60}")
    print(f"Video    : {video_path}")
    print(f"Frames   : {frames_dir}/")
    print(f"Interval : {args.interval}s  |  Max frames: {args.max_frames}")
    print(f"Category : {args.category}")
    print(f"{'='*60}")

    # Step 1a — Extract frames
    print(f"\n[Step 1a] Extracting frames...")
    frames = extract_frames(video_path, frames_dir, args.interval, args.max_frames)
    print(f"  Extracted {len(frames)} frames")

    # Step 1b — Extract audio + transcribe
    print(f"\n[Step 1b] Extracting audio and transcribing...")
    transcript_segments = []
    audio_result = extract_audio(video_path, audio_path)
    if audio_result:
        transcript_segments = transcribe_audio(audio_result)
        print(f"  {len(transcript_segments)} transcript segments ready")
    else:
        print(f"  No audio — transcript will be empty")

    # Steps 2 + 3 — Describe and embed each frame, attach matching transcript
    print()
    frame_results = []
    for i, frame in enumerate(frames, 1):
        print(f"{'─'*60}")
        print(f"Frame {i}/{len(frames)}  |  {frame['frame_file']}  |  t={frame['timestamp']}s")

        matching_segments = segments_at_timestamp(transcript_segments, frame["timestamp"], window=args.interval)
        if matching_segments:
            print(f"  [Transcript] {len(matching_segments)} segment(s) at t={frame['timestamp']}s:")
            for seg in matching_segments:
                print(f"    [{seg['start_time']}s-{seg['end_time']}s] {seg['text']}")
        else:
            print(f"  [Transcript] No audio at t={frame['timestamp']}s")

        print(f"  [Description] Calling vision model...")
        t0 = time.time()
        description = describe_frame(frame["path"])
        t1 = time.time()
        print(f"\n  --- DESCRIPTION ({t1-t0:.1f}s) ---")
        print(f"  {description}\n")

        if description in ("[DESCRIPTION UNAVAILABLE]", "[INVALID FRAME]"):
            print("  Skipping embedding — description failed.\n")
            frame_results.append({
                "frame_file":         frame["frame_file"],
                "timestamp_seconds":  frame["timestamp"],
                "transcript_segments": matching_segments,
                "description":        description,
                "embedding":          None,
            })
            continue

        print(f"  [Embedding] Calling Voyage AI...")
        t2 = time.time()
        vector = embed_text(description)
        t3 = time.time()
        print(f"  --- EMBEDDING ({t3-t2:.1f}s) ---")
        print(f"  Dimension : {len(vector)}")
        print(f"  First 5   : {[round(v, 6) for v in vector[:5]]}")
        print(f"  Min/Max   : {min(vector):.6f} / {max(vector):.6f}\n")

        frame_results.append({
            "frame_file":          frame["frame_file"],
            "timestamp_seconds":   frame["timestamp"],
            "transcript_segments": matching_segments,
            "description":         description,
            "embedding":           vector,
        })

        if i < len(frames):
            print(f"  [Rate limit] Waiting 22s before next frame (Voyage free tier: 3 RPM)...")
            time.sleep(22)

    # Step 4 — Store to MongoDB
    print(f"\n[Step 4] Storing results to MongoDB...")
    store_to_mongodb(video_id, args.category, frame_results, transcript_segments)

    # Save results.json (transcript embeddings now included via in-place mutation)
    results_path = Path(frames_dir) / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "video_id":            video_id,
            "category":            args.category,
            "frames":              frame_results,
            "transcript_segments": transcript_segments,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done. Processed {len(frames)} frames from {Path(video_path).name}")
    print(f"Results saved to: {results_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
