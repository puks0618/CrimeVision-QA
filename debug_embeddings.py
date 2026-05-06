#!/usr/bin/env python3
"""
CrimeVision-QA — Embedding Quality Diagnostic

Identifies why video context descriptions don't match actual video actions.
Tests each stage of the pipeline: Vision → Embed → Store → Search
"""

import argparse
import json
import os
import sys
from pathlib import Path
from pprint import pprint

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm.config import frames_col, transcripts_col, EMBED_PROVIDER, EMBED_DIM
from llm.gen_frame_desc import describe_frame
from llm.get_voyage_embed import embedding_service
from llm.inference import semantic_search_frames


def diagnose_vision_quality(video_id: str, num_samples: int = 3) -> dict:
    """
    STAGE 1: Check if frame descriptions are accurate or hallucinated.
    
    If descriptions are generic/wrong, embeddings will be meaningless.
    """
    print("\n" + "="*70)
    print("STAGE 1: VISION MODEL ACCURACY (gen_frame_desc.py)")
    print("="*70)
    
    # Find sample frames for this video
    docs = list(frames_col.find(
        {"video_id": video_id},
        {"frame_file": 1, "description": 1, "timestamp_seconds": 1},
        limit=num_samples
    ))
    
    if not docs:
        print(f"❌ No frames found for video_id='{video_id}' in MongoDB")
        print("   ACTION: Run process_frames.py to ingest this video first")
        return {"status": "no_data"}
    
    print(f"\n✅ Found {len(docs)} frame documents for '{video_id}'")
    print("\nSample descriptions from MongoDB:")
    print("-" * 70)
    
    for i, doc in enumerate(docs, 1):
        desc = doc.get("description", "[NO DESCRIPTION]")
        timestamp = doc.get("timestamp_seconds", "?")
        frame_file = doc.get("frame_file", "?")
        
        # Check for hallucination markers
        is_hallucinated = any(marker in desc for marker in [
            "[DESCRIPTION UNAVAILABLE]",
            "[INVALID FRAME]",
            "appears to show",
            "generic",
            "unclear",
        ])
        
        status = "⚠️ HALLUCINATED" if is_hallucinated else "✅ OK"
        print(f"\n[Frame {i}] @ {timestamp}s from {frame_file}")
        print(f"Status: {status}")
        print(f"Text: {desc[:150]}...")
    
    # Check if descriptions look specific and factual
    return {
        "status": "ok",
        "sample_docs": docs,
        "num_total": frames_col.count_documents({"video_id": video_id})
    }


def diagnose_embedding_dimension(video_id: str) -> dict:
    """
    STAGE 2: Check if embeddings match MongoDB index dimension.
    
    If dimensions don't match, vector search will fail silently.
    """
    print("\n" + "="*70)
    print("STAGE 2: EMBEDDING DIMENSION (get_voyage_embed.py)")
    print("="*70)
    
    # Get a sample embedding from collection
    sample_doc = frames_col.find_one(
        {"video_id": video_id, "embedding": {"$exists": True}},
        {"embedding": 1}
    )
    
    if not sample_doc:
        print(f"❌ No embeddings found for video_id='{video_id}'")
        return {"status": "no_embeddings"}
    
    embedding = sample_doc.get("embedding", [])
    actual_dim = len(embedding)
    
    print(f"\nEmbedding Provider: {EMBED_PROVIDER.upper()}")
    print(f"Actual Dimension: {actual_dim}")
    
    if EMBED_PROVIDER == "voyage":
        model_name = "voyage-3-large"
    elif EMBED_PROVIDER == "fireworks":
        model_name = "gte-large"
    else:
        model_name = "unknown"
    
    expected_dim = EMBED_DIM
    
    print(f"Expected Dimension: {expected_dim} (for {model_name})")
    
    if actual_dim == expected_dim:
        print(f"✅ Dimensions match!")
    else:
        print(f"❌ DIMENSION MISMATCH!")
        print(f"   If MongoDB index expects {expected_dim}-dim but got {actual_dim}-dim,")
        print(f"   vector search will FAIL SILENTLY or throw index errors.")
    
    # Check for zero-vectors (pollution indicator)
    zero_vectors = frames_col.count_documents({
        "video_id": video_id,
        "embedding": [0.0] * actual_dim
    })
    
    if zero_vectors > 0:
        print(f"\n⚠️ WARNING: Found {zero_vectors} zero-vector embeddings")
        print(f"   These documents will pollute search results")
    
    return {
        "status": "ok",
        "actual_dim": actual_dim,
        "expected_dim": expected_dim,
        "dimension_match": actual_dim == expected_dim,
        "zero_vectors": zero_vectors,
    }


def diagnose_mongodb_index(video_id: str) -> dict:
    """
    STAGE 3: Check if MongoDB vector indexes exist and are active.
    
    If indexes don't exist, $vectorSearch will fail.
    """
    print("\n" + "="*70)
    print("STAGE 3: MONGODB VECTOR INDEXES")
    print("="*70)
    
    try:
        # Get index info
        frame_indexes = frames_col.list_indexes()
        transcript_indexes = transcripts_col.list_indexes()
        
        frame_vector_indexes = [
            idx for idx in frame_indexes 
            if idx.get("key", [None])[0][0] == "vs_frames_index" or 
               "vectorSearch" in str(idx)
        ]
        
        transcript_vector_indexes = [
            idx for idx in transcript_indexes 
            if idx.get("key", [None])[0][0] == "vs_transcripts_index" or 
               "vectorSearch" in str(idx)
        ]
        
        print("\nFrame Collection Indexes:")
        print(f"  Total indexes: {len(list(frame_indexes))}")
        
        if frame_vector_indexes:
            print(f"  ✅ Vector indexes found: {[idx.get('name') for idx in frame_vector_indexes]}")
        else:
            print(f"  ❌ NO VECTOR INDEX found for frames!")
            print(f"     ACTION: Create 'vs_frames_index' in MongoDB Atlas console")
        
        print("\nTranscript Collection Indexes:")
        print(f"  Total indexes: {len(list(transcript_indexes))}")
        
        if transcript_vector_indexes:
            print(f"  ✅ Vector indexes found: {[idx.get('name') for idx in transcript_vector_indexes]}")
        else:
            print(f"  ❌ NO VECTOR INDEX found for transcripts!")
            print(f"     ACTION: Create 'vs_transcripts_index' in MongoDB Atlas console")
        
        return {"status": "ok", "frame_indexes_exist": bool(frame_vector_indexes)}
        
    except Exception as e:
        print(f"❌ Error checking indexes: {e}")
        return {"status": "error", "error": str(e)}


def diagnose_vector_search(video_id: str, test_query: str) -> dict:
    """
    STAGE 4: Test vector search to see if it returns relevant results.
    
    If search returns wrong results, embeddings or index is misconfigured.
    """
    print("\n" + "="*70)
    print("STAGE 4: VECTOR SEARCH QUALITY (inference.py)")
    print("="*70)
    
    print(f"\nTest Query: '{test_query}'")
    print(f"Video ID: {video_id}")
    
    try:
        results = semantic_search_frames(test_query, video_id=video_id, k=5)
        
        if not results:
            print("\n❌ VECTOR SEARCH RETURNED NO RESULTS")
            print("   Possible causes:")
            print("   1. Vector index not created in MongoDB Atlas")
            print("   2. No frames ingested for this video")
            print("   3. Embedding dimensions don't match index")
            return {"status": "no_results"}
        
        print(f"\n✅ Found {len(results)} results")
        print("\nTop Results:")
        print("-" * 70)
        
        for i, result in enumerate(results[:3], 1):
            score = result.get("score", 0)
            desc = result.get("description", "?")
            timestamp = result.get("timestamp_seconds", "?")
            
            print(f"\n[{i}] Score: {score:.4f} @ {timestamp}s")
            print(f"    Description: {desc[:100]}...")
        
        return {
            "status": "ok",
            "num_results": len(results),
            "top_scores": [r.get("score", 0) for r in results[:3]],
            "results": results
        }
        
    except Exception as e:
        print(f"❌ Vector search failed: {e}")
        return {"status": "error", "error": str(e)}


def diagnose_description_to_query_match(video_id: str) -> dict:
    """
    STAGE 5: Check if frame descriptions match the actual visual content.
    
    This is the critical quality check.
    """
    print("\n" + "="*70)
    print("STAGE 5: DESCRIPTION QUALITY vs ACTUAL VIDEO CONTENT")
    print("="*70)
    
    print("\n⚠️ MANUAL INSPECTION REQUIRED:")
    print("-" * 70)
    print(f"\n1. Open the video frames in: frames/{video_id}/")
    print(f"2. Look at frame descriptions in MongoDB for {video_id}")
    print(f"3. Compare:")
    print(f"   ✓ Do descriptions accurately describe what you SEE?")
    print(f"   ✗ Are descriptions generic/hallucinated?")
    print(f"   ✗ Are important details missing?")
    
    # Get sample docs for manual inspection
    docs = list(frames_col.find(
        {"video_id": video_id},
        {"frame_file": 1, "description": 1, "timestamp_seconds": 1},
        limit=5
    ))
    
    print(f"\n4. Sample frames for inspection:")
    for doc in docs:
        print(f"\n   Frame: {doc.get('frame_file')}")
        print(f"   Description: {doc.get('description')}")
    
    return {"status": "manual_inspection_needed"}


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose CrimeVision embedding quality issues"
    )
    parser.add_argument("--video-id", required=True, help="Video ID to diagnose")
    parser.add_argument("--query", default="person running", help="Test query for vector search")
    parser.add_argument("--output", help="Save diagnostic report to JSON")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("CrimeVision-QA — EMBEDDING DIAGNOSTIC REPORT")
    print("="*70)
    print(f"\nVideo ID: {args.video_id}")
    print(f"Embedding Provider: {EMBED_PROVIDER}")
    
    report = {}
    
    # Run all diagnostic stages
    report["stage1_vision"] = diagnose_vision_quality(args.video_id)
    report["stage2_dimension"] = diagnose_embedding_dimension(args.video_id)
    report["stage3_indexes"] = diagnose_mongodb_index(args.video_id)
    report["stage4_search"] = diagnose_vector_search(args.video_id, args.query)
    report["stage5_quality"] = diagnose_description_to_query_match(args.video_id)
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    
    all_ok = all(
        report.get(f"stage{i}", {}).get("status") in ["ok", "manual_inspection_needed"]
        for i in range(1, 6)
    )
    
    if all_ok:
        print("\n✅ All checks passed (or require manual verification)")
        print("\nIf vector search is still returning wrong results:")
        print("  → The issue is DESCRIPTION QUALITY (Stage 1)")
        print("  → Check if vision model (kimi-k2p5) is generating accurate descriptions")
        print("  → Consider re-ingesting with different vision model")
    else:
        print("\n❌ Issues found. Review stages above for detailed diagnostics.")
    
    # Save report
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Report saved to: {args.output}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
