"""
Creates the two Atlas vector search indexes required for CrimeVision-QA.
Run once before querying. Safe to re-run — skips indexes that already exist.

Usage:
    python setup_indexes.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv(Path(__file__).parent / ".env")

MONGODB_URI = os.environ["MONGODB_URI"]
DB_NAME = os.getenv("MONGODB_DB_NAME", "video_intelligence")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm.config import EMBED_DIM

INDEXES = [
    {
        "collection": "video_intelligence",
        "name": "vs_frames_index",
        "definition": {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": EMBED_DIM,
                    "similarity": "cosine",
                },
                {
                    "type": "filter",
                    "path": "video_id",
                },
            ]
        },
    },
    {
        "collection": "video_intelligence_transcripts",
        "name": "vs_transcripts_index",
        "definition": {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": EMBED_DIM,
                    "similarity": "cosine",
                },
                {
                    "type": "filter",
                    "path": "video_id",
                },
            ]
        },
    },
]


def existing_index_names(collection) -> set[str]:
    try:
        return {idx["name"] for idx in collection.list_search_indexes()}
    except Exception:
        return set()


def main() -> None:
    print(f"\n[Setup] Connecting to MongoDB...")
    client = MongoClient(MONGODB_URI)
    client.admin.command("ping")
    print(f"[Setup] Connected.")

    db = client[DB_NAME]

    # Ensure collections exist before creating indexes (Atlas requires this)
    existing_collections = db.list_collection_names()
    for idx in INDEXES:
        if idx["collection"] not in existing_collections:
            print(f"[Setup] Creating collection '{idx['collection']}'...")
            db.create_collection(idx["collection"])

    for idx in INDEXES:
        col = db[idx["collection"]]
        existing = existing_index_names(col)

        if idx["name"] in existing:
            print(f"[Setup] Dropping existing index '{idx['name']}' to recreate with updated definition...")
            col.drop_search_index(idx["name"])
            time.sleep(5)  # give Atlas a moment to remove it

        print(f"[Setup] Creating index '{idx['name']}' on '{idx['collection']}'...")
        col.create_search_index({"name": idx["name"], "definition": idx["definition"], "type": "vectorSearch"})
        print(f"[Setup] Created. (Atlas takes ~30s to build it in the background)")

    # Create regular text indexes for $text keyword search (hybrid retrieval)
    text_indexes = [
        ("video_intelligence",             "description"),
        ("video_intelligence_transcripts", "text"),
    ]
    for col_name, field in text_indexes:
        col = db[col_name]
        existing_regular = {v["name"] for v in col.list_indexes()}
        index_name = f"{field}_text"
        if index_name not in existing_regular:
            print(f"[Setup] Creating text index on '{col_name}.{field}'...")
            col.create_index([(field, "text")], name=index_name)
            print(f"[Setup] Text index created.")
        else:
            print(f"[Setup] Text index on '{col_name}.{field}' already exists — skipping.")

    print(f"\n[Setup] Waiting 35s for vector indexes to become active...")
    time.sleep(35)

    print(f"[Setup] Done. All indexes are ready.\n")
    client.close()


if __name__ == "__main__":
    main()
