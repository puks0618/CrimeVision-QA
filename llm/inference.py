from __future__ import annotations
"""
CrimeVision-QA — Vector Semantic Search

Pure $vectorSearch aggregation against the Atlas vector indexes.
Requires the two vector indexes to be created in the Atlas console:
  - vs_frames_index      on video_intelligence.embedding
  - vs_transcripts_index on video_intelligence_transcripts.embedding
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm.config import frames_col, transcripts_col
from llm.get_voyage_embed import embedding_service


def semantic_search_frames(
    query: str,
    video_id: str | None = None,
    k: int = 5,
    num_candidates: int = 100,
) -> list[dict]:
    """Search frame descriptions by semantic similarity.

    Returns up to *k* results sorted by descending vector score.
    Returns [] immediately if no frames exist (avoids API call).
    Each result includes: video_id, frame_file, timestamp_seconds,
    description, category, score.
    """
    # Short-circuit: skip embedding call if frames collection has no docs for this video
    filter_check: dict = {}
    if video_id:
        filter_check["video_id"] = video_id
    if frames_col.count_documents(filter_check, limit=1) == 0:
        return []

    query_vec = embedding_service.embed_single(query)

    vector_search_stage: dict = {
        "index": "vs_frames_index",
        "path": "embedding",
        "queryVector": query_vec,
        "numCandidates": num_candidates,
        "limit": k,
    }
    if video_id:
        vector_search_stage["filter"] = {"video_id": {"$eq": video_id}}

    pipeline = [
        {"$vectorSearch": vector_search_stage},
        {
            "$project": {
                "_id": 1,
                "video_id": 1,
                "frame_file": 1,
                "frame_number": 1,
                "timestamp_seconds": 1,
                "description": 1,
                "category": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    results = list(frames_col.aggregate(pipeline))
    return results


def semantic_search_transcripts(
    query: str,
    video_id: str | None = None,
    k: int = 5,
    num_candidates: int = 100,
) -> list[dict]:
    """Search transcript segments by semantic similarity.

    Returns [] immediately if no transcripts exist (avoids API call).
    """
    # Short-circuit: skip embedding call if transcripts collection is empty
    filter_check: dict = {}
    if video_id:
        filter_check["video_id"] = video_id
    if transcripts_col.count_documents(filter_check, limit=1) == 0:
        return []

    query_vec = embedding_service.embed_single(query)

    vector_search_stage: dict = {
        "index": "vs_transcripts_index",
        "path": "embedding",
        "queryVector": query_vec,
        "numCandidates": num_candidates,
        "limit": k,
    }
    if video_id:
        vector_search_stage["filter"] = {"video_id": {"$eq": video_id}}

    pipeline = [
        {"$vectorSearch": vector_search_stage},
        {
            "$project": {
                "_id": 1,
                "video_id": 1,
                "segment_index": 1,
                "start_time": 1,
                "end_time": 1,
                "text": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    return list(transcripts_col.aggregate(pipeline))


def semantic_search_all(
    query: str,
    video_id: str | None = None,
    k: int = 5,
) -> dict:
    """Search both frames and transcripts, return combined dict."""
    frames = semantic_search_frames(query, video_id=video_id, k=k)
    transcripts = semantic_search_transcripts(query, video_id=video_id, k=k)
    return {"frames": frames, "transcripts": transcripts}
