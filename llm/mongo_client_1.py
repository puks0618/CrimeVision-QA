from __future__ import annotations
"""
CrimeVision-QA — MongoDB Collection & Index Setup

Creates collections and regular indexes.  Vector search indexes must be
created manually via the Atlas console UI (M0 free tier limitation).

Run directly to initialise the database:
    python llm/mongo_client_1.py
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running as script from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymongo import ASCENDING, TEXT
from pymongo.errors import OperationFailure

from llm.config import (
    DB_NAME,
    EMBED_DIM,
    frames_col,
    transcripts_col,
    video_library_col,
    incidents_col,
    get_db,
)

# ---------------------------------------------------------------------------
# Regular MongoDB indexes (work on M0 free tier)
# ---------------------------------------------------------------------------

_INDEXES = {
    "video_intelligence": [
        # Text index for keyword search (used in hybrid retrieval)
        {"keys": [("description", TEXT)], "name": "text_description"},
        # Compound index for time-ordered lookups within a video
        {
            "keys": [("video_id", ASCENDING), ("timestamp_seconds", ASCENDING)],
            "name": "video_timestamp",
        },
        # Category filter
        {"keys": [("category", ASCENDING)], "name": "category"},
    ],
    "video_intelligence_transcripts": [
        # Text index for keyword search
        {"keys": [("text", TEXT)], "name": "text_transcript"},
        # Compound index for time-ordered lookups within a video
        {
            "keys": [("video_id", ASCENDING), ("start_time", ASCENDING)],
            "name": "video_start_time",
        },
    ],
    "video_library": [
        # Unique index on video_id
        {
            "keys": [("video_id", ASCENDING)],
            "name": "video_id_unique",
            "unique": True,
        },
    ],
    "previous_frame_incidents": [
        {
            "keys": [("video_id", ASCENDING), ("timestamp_seconds", ASCENDING)],
            "name": "incident_video_time",
        },
    ],
}


def create_regular_indexes() -> None:
    """Create all regular MongoDB indexes. Safe to call multiple times."""
    db = get_db()
    for collection_name, index_defs in _INDEXES.items():
        col = db[collection_name]
        existing = {idx["name"] for idx in col.list_indexes()}
        for idx_def in index_defs:
            name = idx_def["name"]
            if name in existing:
                print(f"  [skip] {collection_name}.{name} (already exists)")
                continue
            try:
                col.create_index(
                    idx_def["keys"],
                    name=name,
                    unique=idx_def.get("unique", False),
                )
                print(f"  [created] {collection_name}.{name}")
            except OperationFailure as exc:
                print(f"  [error] {collection_name}.{name}: {exc}")


# ---------------------------------------------------------------------------
# Vector search index definitions (Atlas console — manual creation)
# ---------------------------------------------------------------------------

_VECTOR_INDEX_INSTRUCTIONS = """
==========================================================================
  MANUAL STEP REQUIRED — Atlas Vector Search Indexes
==========================================================================

M0 free tier does not support programmatic vector index creation.
Go to your Atlas console and create these indexes manually:

1. Navigate to: Atlas Console -> Database -> Browse Collections
2. Select collection -> Search Indexes tab -> Create Index
3. Choose "JSON Editor" and paste the definition below.

--- Index 1: vs_frames_index ---
Collection: {db_name}.video_intelligence
Name:       vs_frames_index

{{
  "fields": [
    {{
      "type": "vector",
      "path": "embedding",
      "numDimensions": {embed_dim},
      "similarity": "cosine"
    }}
  ]
}}

--- Index 2: vs_transcripts_index ---
Collection: {db_name}.video_intelligence_transcripts
Name:       vs_transcripts_index

{{
  "fields": [
    {{
      "type": "vector",
      "path": "embedding",
      "numDimensions": {embed_dim},
      "similarity": "cosine"
    }}
  ]
}}

==========================================================================
  After creating both indexes, wait ~1 minute for them to become ACTIVE
  before running any queries.
==========================================================================
"""


# ---------------------------------------------------------------------------
# Helper: insert a sample document (useful for index creation validation)
# ---------------------------------------------------------------------------

def _ensure_collections_exist() -> None:
    """Touch each collection so they show up in Atlas console."""
    db = get_db()
    for col_name in _INDEXES:
        if col_name not in db.list_collection_names():
            db.create_collection(col_name)
            print(f"  [created collection] {col_name}")
        else:
            print(f"  [exists] collection {col_name}")


# ---------------------------------------------------------------------------
# Main — run setup
# ---------------------------------------------------------------------------

def setup_database() -> None:
    """Run full database setup: collections + regular indexes + instructions."""
    print(f"\n[MongoDB Setup] Database: {DB_NAME}")
    print("-" * 50)

    print("\nCollections:")
    _ensure_collections_exist()

    print("\nRegular Indexes:")
    create_regular_indexes()

    print(_VECTOR_INDEX_INSTRUCTIONS.format(db_name=DB_NAME, embed_dim=EMBED_DIM))


if __name__ == "__main__":
    setup_database()
