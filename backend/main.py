from __future__ import annotations
"""
CrimeVision-QA — FastAPI Backend

Endpoints:
  POST /api/chat              — Main QA query
  GET  /api/videos            — List processed videos
  GET  /api/videos/{id}/frames — Frame metadata for a video
  GET  /api/health            — System health check

Static mounts:
  /videos/*  → ./videos/
  /frames/*  → ./frames/

Run:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

import asyncio
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Trigger config validation at startup
from llm.config import (
    EMBED_PROVIDER,
    REASONER_PROVIDER,
    frames_col,
    video_library_col,
)
from llm.agent import run_agent

app = FastAPI(
    title="CrimeVision-QA",
    description="Multimodal RAG for Surveillance Video Q&A",
    version="1.0.0",
)

# One pipeline at a time (Voyage rate limits make parallelism pointless)
_executor = ThreadPoolExecutor(max_workers=1)
_jobs: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static file serving (videos + frames)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VIDEOS_DIR = _PROJECT_ROOT / "videos"
_FRAMES_DIR = _PROJECT_ROOT / "frames"

_VIDEOS_DIR.mkdir(exist_ok=True)
_FRAMES_DIR.mkdir(exist_ok=True)

app.mount("/videos", StaticFiles(directory=str(_VIDEOS_DIR)), name="videos")
app.mount("/frames", StaticFiles(directory=str(_FRAMES_DIR)), name="frames")

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    video_id: str
    strategy: str = "zero_shot"  # zero_shot | cot | few_shot | react


class SourceDoc(BaseModel):
    frame_file: Optional[str] = None
    timestamp_seconds: Optional[float] = None
    description: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    text: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    timestamps: list[float]
    sources: list[dict]
    strategy_used: str
    processing_time: float


class VideoInfo(BaseModel):
    video_id: str
    filename: Optional[str] = None
    video_url: Optional[str] = None
    category: Optional[str] = None
    duration_seconds: Optional[float] = None
    frame_count: Optional[int] = None
    transcript_segments: Optional[int] = None


class FrameInfo(BaseModel):
    frame_file: str
    frame_number: int
    timestamp_seconds: float
    description: str


class UploadResponse(BaseModel):
    job_id: str
    video_id: str
    message: str


class JobStatus(BaseModel):
    job_id: str
    video_id: str
    status: str  # queued | extracting | describing | embedding | transcribing | storing | done | error
    progress: int
    message: str
    error: Optional[str] = None
    frame_count: Optional[int] = None


_VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def _find_video_url(video_id: str) -> Optional[str]:
    for ext in _VIDEO_EXTENSIONS:
        if (_VIDEOS_DIR / f"{video_id}{ext}").exists():
            return f"/videos/{video_id}{ext}"
    return None


def _run_pipeline(job_id: str, video_path: Path, video_id: str, category: str) -> None:
    """Runs in a thread-pool thread. Updates _jobs[job_id] as it progresses."""
    sys.path.insert(0, str(_PROJECT_ROOT))
    from test_pipeline import (  # noqa: PLC0415
        describe_frame,
        embed_text,
        extract_audio,
        extract_frames,
        store_to_mongodb,
        transcribe_audio,
    )

    def update(status: str, progress: int, message: str, **kw: Any) -> None:
        _jobs[job_id].update({"status": status, "progress": progress, "message": message, **kw})

    try:
        # Step 1 — extract frames
        update("extracting", 5, "Extracting frames…")
        frames_dir = _PROJECT_ROOT / "frames" / video_id
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_list = extract_frames(str(video_path), str(frames_dir), interval_seconds=2, max_frames=0)
        # store_to_mongodb expects "timestamp_seconds"; extract_frames returns "timestamp"
        for f in frame_list:
            f["timestamp_seconds"] = f["timestamp"]
        total = max(len(frame_list), 1)
        update("extracting", 10, f"Extracted {len(frame_list)} frames")

        # Step 2 — describe each frame (10%→55%)
        for i, f in enumerate(frame_list):
            pct = 10 + int((i / total) * 45)
            update("describing", pct, f"Describing frame {i + 1}/{len(frame_list)}…")
            f["description"] = describe_frame(f["path"])

        # Step 3 — embed each frame (55%→85%)
        for i, f in enumerate(frame_list):
            pct = 55 + int((i / total) * 30)
            update("embedding", pct, f"Embedding frame {i + 1}/{len(frame_list)}…")
            f["embedding"] = embed_text(f.get("description", ""))

        # Step 4 — audio transcription (85%→90%)
        update("transcribing", 85, "Transcribing audio…")
        audio_path = frames_dir / "audio.mp3"
        extract_audio(str(video_path), str(audio_path))
        transcript_segments = transcribe_audio(str(audio_path)) if audio_path.exists() else []

        # Step 5 — store to MongoDB (90%→100%)
        update("storing", 90, "Storing to MongoDB…")
        store_to_mongodb(video_id, category, frame_list, transcript_segments)
        _jobs[job_id]["frame_count"] = len(frame_list)
        update("done", 100, f"Done — {len(frame_list)} frames indexed")

    except Exception as exc:
        _jobs[job_id].update({
            "status": "error", "progress": 0,
            "message": "Pipeline failed", "error": str(exc),
        })


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Run RAG query on a video and return a timestamped answer."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if not request.video_id.strip():
        raise HTTPException(status_code=400, detail="video_id cannot be empty")

    valid_strategies = {"zero_shot", "cot", "few_shot", "react"}
    if request.strategy not in valid_strategies:
        raise HTTPException(
            status_code=400,
            detail=f"strategy must be one of {valid_strategies}",
        )

    start_time = time.perf_counter()

    result = await run_agent(
        query=request.message,
        video_id=request.video_id,
        strategy=request.strategy,
    )

    elapsed = round(time.perf_counter() - start_time, 2)

    # Convert any non-serializable MongoDB fields (ObjectId, etc.) in sources
    raw_sources = result.get("sources", [])
    clean_sources: list[dict] = []
    for doc in raw_sources:
        clean = {
            k: str(v) if hasattr(v, "__class__") and v.__class__.__name__ == "ObjectId" else v
            for k, v in doc.items()
            if k != "_id"  # drop internal Mongo _id
        }
        clean_sources.append(clean)

    return ChatResponse(
        answer=result["answer"],
        timestamps=result.get("timestamps", []),
        sources=clean_sources,
        strategy_used=result.get("strategy_used", request.strategy),
        processing_time=elapsed,
    )


@app.get("/api/videos", response_model=list[VideoInfo])
async def list_videos() -> list[VideoInfo]:
    """Return all videos that have been processed into the database."""
    docs = list(video_library_col.find({}, {"_id": 0}).limit(200))
    infos = []
    for d in docs:
        d["video_url"] = _find_video_url(d["video_id"])
        infos.append(VideoInfo(**d))
    return infos


@app.get("/api/videos/{video_id}/frames", response_model=list[FrameInfo])
async def get_video_frames(video_id: str) -> list[FrameInfo]:
    """Return all frame metadata for a specific video."""
    docs = list(
        frames_col.find(
            {"video_id": video_id},
            {"_id": 0, "frame_file": 1, "frame_number": 1, "timestamp_seconds": 1, "description": 1},
        ).sort("timestamp_seconds", 1)
    )
    if not docs:
        raise HTTPException(status_code=404, detail=f"No frames found for video_id '{video_id}'")
    return [FrameInfo(**d) for d in docs]


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    """Return system health and active provider configuration."""
    return {
        "status": "ok",
        "embed_provider": EMBED_PROVIDER,
        "reasoner_provider": REASONER_PROVIDER,
    }


@app.post("/api/upload", response_model=UploadResponse)
async def upload_video(
    file: UploadFile = File(...),
    category: str = Form("Unknown"),
) -> UploadResponse:
    """Accept a video upload, save it, and kick off the ingestion pipeline."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in _VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}")

    video_id = Path(file.filename).stem
    dest = _VIDEOS_DIR / file.filename

    contents = await file.read()
    dest.write_bytes(contents)

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "video_id": video_id,
        "status": "queued",
        "progress": 0,
        "message": "Queued for processing",
        "error": None,
        "frame_count": None,
    }
    _executor.submit(_run_pipeline, job_id, dest, video_id, category)

    return UploadResponse(job_id=job_id, video_id=video_id, message="Upload received, processing started")


@app.get("/api/upload/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str) -> JobStatus:
    """Poll processing progress for an upload job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(**_jobs[job_id])
