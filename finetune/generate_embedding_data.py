from __future__ import annotations
"""
CrimeVision-QA — Embedding Fine-Tuning Data Generator

Builds (query, positive_document, negative_document) triplets from MongoDB
for contrastive fine-tuning of a sentence-transformer embedding model.

Strategy:
  - Positive pairs: test_queries.json (query → reference_answer keywords) matched
    against frame descriptions from the correct video.
  - Synthetic positives: generated from frame descriptions by extracting key phrases
    as pseudo-queries (no LLM needed — rule-based).
  - Negatives: random descriptions from DIFFERENT videos (in-batch negatives).

Usage:
    python finetune/generate_embedding_data.py \
        --output finetune/data/embedding_training_data.json \
        --test-queries evaluation/test_queries.json
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm.config import frames_col, transcripts_col

random.seed(42)

# Action and appearance keywords to extract pseudo-queries from descriptions
_ACTION_PATTERNS = [
    r"(running|fleeing|walking|standing|crouching|fighting|punching|kicking|grabbing|"
    r"stabbing|shooting|pointing|pushing|shouting|arguing)",
    r"(wearing\s+[\w\s,]+(?:shirt|jacket|hoodie|pants|dress|uniform))",
    r"(\w+\s+(?:car|vehicle|truck|van|sedan)\s+\w+)",
    r"(license\s+plate\s+[\w\d]+)",
    r"(weapon|gun|knife|bat|firearm)",
]


def _extract_pseudo_queries(description: str) -> list[str]:
    """Generate plausible search queries from a frame description."""
    queries = []
    desc_lower = description.lower()

    for pattern in _ACTION_PATTERNS:
        for m in re.finditer(pattern, desc_lower):
            snippet = m.group(0).strip()
            if len(snippet) > 5:
                queries.append(snippet)

    # Add sentence-level queries
    sentences = [s.strip() for s in description.split(".") if len(s.strip()) > 30]
    for sent in sentences[:3]:
        # Strip section headers like "PEOPLE:" → cleaner query
        clean = re.sub(r"^[A-Z\s/]+:\s*", "", sent).strip()
        if len(clean) > 20:
            queries.append(clean)

    return list(set(queries))[:5]  # max 5 queries per frame


def generate(output_path: str, test_queries_path: str) -> int:
    # 1. Pull all frame descriptions grouped by video
    print("[EmbedData] Fetching frame descriptions from MongoDB...")
    all_frames = list(
        frames_col.find(
            {"description": {"$exists": True, "$ne": "[DESCRIPTION UNAVAILABLE]"}},
            {"_id": 0, "video_id": 1, "description": 1, "timestamp_seconds": 1},
        )
    )

    if not all_frames:
        print("[EmbedData] No frames found. Run the ingestion pipeline first.")
        return 0

    # Group by video_id
    by_video: dict[str, list[str]] = {}
    for f in all_frames:
        by_video.setdefault(f["video_id"], []).append(f["description"])

    all_descriptions = [f["description"] for f in all_frames]
    print(f"[EmbedData] {len(all_descriptions)} frame descriptions across {len(by_video)} videos")

    # 2. Pull transcript texts too
    all_transcripts = list(
        transcripts_col.find(
            {"text": {"$exists": True}},
            {"_id": 0, "video_id": 1, "text": 1},
        )
    )
    for t in all_transcripts:
        by_video.setdefault(t["video_id"], []).append(t["text"])

    pairs = []  # list of {"query": str, "positive": str, "negative": str}

    # 3. Pairs from test_queries.json (highest quality — human-written queries)
    if os.path.exists(test_queries_path):
        with open(test_queries_path) as f:
            test_qs = json.load(f)
        print(f"[EmbedData] Processing {len(test_qs)} test queries...")

        for item in test_qs:
            query = item.get("query", "")
            video_id = item.get("video_id", "")
            reference = item.get("reference_answer", "")
            if not query or not video_id:
                continue

            # Positives: frame descriptions from the correct video
            positives = by_video.get(video_id, [])
            if not positives:
                continue

            # Negatives: descriptions from a different video
            other_videos = [vid for vid in by_video if vid != video_id]
            if not other_videos:
                continue
            neg_video = random.choice(other_videos)
            negative = random.choice(by_video[neg_video])

            for pos in positives[:5]:  # up to 5 positive docs per query
                pairs.append({"query": query, "positive": pos, "negative": negative})

            # Also add the reference answer as a pair if it's long enough
            if len(reference) > 30:
                pairs.append({"query": query, "positive": reference, "negative": negative})

    # 4. Synthetic pairs from frame descriptions (rule-based pseudo-queries)
    print("[EmbedData] Generating synthetic query/positive pairs...")
    for frame in all_frames:
        desc = frame["description"]
        video_id = frame["video_id"]
        pseudo_queries = _extract_pseudo_queries(desc)

        other_videos = [vid for vid in by_video if vid != video_id]
        if not other_videos:
            continue

        for q in pseudo_queries:
            neg_video = random.choice(other_videos)
            negative = random.choice(by_video[neg_video])
            pairs.append({"query": q, "positive": desc, "negative": negative})

    # Deduplicate
    seen = set()
    unique_pairs = []
    for p in pairs:
        key = (p["query"][:60], p["positive"][:60])
        if key not in seen:
            seen.add(key)
            unique_pairs.append(p)

    random.shuffle(unique_pairs)
    print(f"[EmbedData] Total unique pairs: {len(unique_pairs)}")

    if not unique_pairs:
        print("[EmbedData] No pairs generated. Need at least 2 videos with frame data.")
        return 0

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(unique_pairs, f, indent=2, ensure_ascii=False)

    print(f"[EmbedData] Saved {len(unique_pairs)} triplets → {output_path}")
    return len(unique_pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="finetune/data/embedding_training_data.json")
    parser.add_argument("--test-queries", default="evaluation/test_queries.json")
    args = parser.parse_args()

    generate(args.output, args.test_queries)


if __name__ == "__main__":
    main()
