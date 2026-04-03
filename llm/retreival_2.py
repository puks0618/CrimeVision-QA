from __future__ import annotations
"""
CrimeVision-QA — Hybrid Retrieval with Manual RRF

Combines vector search ($vectorSearch) with regular MongoDB $text search
using Reciprocal Rank Fusion (RRF).

Design decisions:
- Always uses manual RRF — no $rankFusion dependency (works on M0 free tier).
- $text indexes are regular MongoDB indexes (no Atlas Search required).
- Weights: 70% vector, 30% keyword by default.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm.config import frames_col, transcripts_col
from llm.inference import semantic_search_frames, semantic_search_transcripts

_RRF_K = 60  # RRF constant (higher = less aggressive rank penalty)


def _reciprocal_rank_fusion(
    vector_results: list[dict],
    text_results: list[dict],
    vector_weight: float = 0.7,
    text_weight: float = 0.3,
) -> list[dict]:
    """Combine two ranked lists using Reciprocal Rank Fusion.

    Formula: score(d) = Σ weight_i * (1 / (RRF_K + rank_i))
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for rank, doc in enumerate(vector_results):
        doc_id = str(doc["_id"])
        scores[doc_id] = vector_weight / (_RRF_K + rank + 1)
        doc_map[doc_id] = doc

    for rank, doc in enumerate(text_results):
        doc_id = str(doc["_id"])
        rrf_score = text_weight / (_RRF_K + rank + 1)
        scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score
        if doc_id not in doc_map:
            doc_map[doc_id] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"rrf_score": score, **doc_map[doc_id]} for doc_id, score in ranked]


def _text_search_frames(
    query: str,
    video_id: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Keyword search on frame descriptions using MongoDB $text operator."""
    match_filter: dict = {"$text": {"$search": query}}
    if video_id:
        match_filter["video_id"] = video_id

    pipeline = [
        {"$match": match_filter},
        {"$addFields": {"text_score": {"$meta": "textScore"}}},
        {"$sort": {"text_score": -1}},
        {"$limit": limit},
        {
            "$project": {
                "_id": 1,
                "video_id": 1,
                "frame_file": 1,
                "frame_number": 1,
                "timestamp_seconds": 1,
                "description": 1,
                "category": 1,
                "text_score": 1,
            }
        },
    ]
    return list(frames_col.aggregate(pipeline))


def _text_search_transcripts(
    query: str,
    video_id: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Keyword search on transcripts using MongoDB $text operator."""
    match_filter: dict = {"$text": {"$search": query}}
    if video_id:
        match_filter["video_id"] = video_id

    pipeline = [
        {"$match": match_filter},
        {"$addFields": {"text_score": {"$meta": "textScore"}}},
        {"$sort": {"text_score": -1}},
        {"$limit": limit},
        {
            "$project": {
                "_id": 1,
                "video_id": 1,
                "segment_index": 1,
                "start_time": 1,
                "end_time": 1,
                "text": 1,
                "text_score": 1,
            }
        },
    ]
    return list(transcripts_col.aggregate(pipeline))


def hybrid_search_frames(
    query: str,
    video_id: str | None = None,
    k: int = 5,
    vector_weight: float = 0.7,
    text_weight: float = 0.3,
) -> list[dict]:
    """Hybrid vector + keyword search on frame descriptions.

    Returns top-*k* results fused by RRF.
    """
    vector_results = semantic_search_frames(query, video_id=video_id, k=k * 3)
    text_results = _text_search_frames(query, video_id=video_id, limit=k * 3)
    fused = _reciprocal_rank_fusion(vector_results, text_results, vector_weight, text_weight)
    return fused[:k]


def hybrid_search_transcripts(
    query: str,
    video_id: str | None = None,
    k: int = 5,
    vector_weight: float = 0.7,
    text_weight: float = 0.3,
) -> list[dict]:
    """Hybrid vector + keyword search on transcript segments.

    Returns top-*k* results fused by RRF.
    """
    vector_results = semantic_search_transcripts(query, video_id=video_id, k=k * 3)
    text_results = _text_search_transcripts(query, video_id=video_id, limit=k * 3)
    fused = _reciprocal_rank_fusion(vector_results, text_results, vector_weight, text_weight)
    return fused[:k]


def time_windowed_search(
    query: str,
    video_id: str,
    start_time: float,
    end_time: float,
    k: int = 10,
) -> dict:
    """Search within a specific time window in both frames and transcripts."""
    # Frames within time window
    frame_filter = {
        "video_id": video_id,
        "timestamp_seconds": {"$gte": start_time, "$lte": end_time},
    }
    frames = list(
        frames_col.find(
            frame_filter,
            {
                "_id": 0,
                "frame_file": 1,
                "timestamp_seconds": 1,
                "description": 1,
                "category": 1,
            },
        ).sort("timestamp_seconds", 1).limit(k)
    )

    # Transcripts overlapping with time window
    transcript_filter = {
        "video_id": video_id,
        "start_time": {"$lte": end_time},
        "end_time": {"$gte": start_time},
    }
    transcripts = list(
        transcripts_col.find(
            transcript_filter,
            {"_id": 0, "start_time": 1, "end_time": 1, "text": 1},
        ).sort("start_time", 1).limit(k)
    )

    return {
        "frames": frames,
        "transcripts": transcripts,
        "time_range": {"start": start_time, "end": end_time},
    }
