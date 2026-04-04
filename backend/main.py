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
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
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
    category: Optional[str] = None
    duration_seconds: Optional[float] = None
    frame_count: Optional[int] = None
    transcript_segments: Optional[int] = None


class FrameInfo(BaseModel):
    frame_file: str
    frame_number: int
    timestamp_seconds: float
    description: str


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
    return [VideoInfo(**d) for d in docs]


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
