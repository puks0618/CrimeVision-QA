#!/usr/bin/env python3
"""
CrimeVision-QA — MongoDB Cleanup Script

Removes corrupted embeddings and duplicates from previous ingestions.
Run this BEFORE re-ingesting videos with improved prompts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm.config import frames_col, transcripts_col, DB_NAME


def cleanup_zero_vectors():
    """Remove zero-vector embeddings that pollute the search index."""
    print("\n" + "="*70)
    print("CLEANING ZERO-VECTORS FROM FRAMES")
    print("="*70)
    
    # Build query for zero vectors (1024-dim)
    zero_vector = [0.0] * 1024
    
    result = frames_col.delete_many({"embedding": {"$eq": zero_vector}})
    print(f"✅ Deleted {result.deleted_count} zero-vector frame documents")
    
    print("\n" + "="*70)
    print("CLEANING ZERO-VECTORS FROM TRANSCRIPTS")
    print("="*70)
    
    result = transcripts_col.delete_many({"embedding": {"$eq": zero_vector}})
    print(f"✅ Deleted {result.deleted_count} zero-vector transcript documents")


def cleanup_duplicates():
    """Remove duplicate frame documents keeping only the most recent."""
    print("\n" + "="*70)
    print("CLEANING DUPLICATES FROM FRAMES")
    print("="*70)
    
    # Find duplicates: same video_id + frame_file
    pipeline = [
        {
            "$group": {
                "_id": {"video_id": "$video_id", "frame_file": "$frame_file"},
                "count": {"$sum": 1},
                "ids": {"$push": "$_id"},
                "created_at": {"$push": "$created_at"}
            }
        },
        {"$match": {"count": {"$gt": 1}}}
    ]
    
    duplicates = list(frames_col.aggregate(pipeline))
    
    if not duplicates:
        print("✅ No duplicates found")
        return
    
    total_deleted = 0
    for dup in duplicates:
        ids = dup["ids"]
        created_ats = dup["created_at"]
        
        # Keep the most recent (highest timestamp)
        keep_idx = created_ats.index(max(created_ats))
        ids_to_delete = ids[:keep_idx] + ids[keep_idx+1:]
        
        result = frames_col.delete_many({"_id": {"$in": ids_to_delete}})
        total_deleted += result.deleted_count
        
        print(f"  Cleaned {dup['_id']}: kept 1, deleted {len(ids_to_delete)}")
    
    print(f"\n✅ Total duplicate documents deleted: {total_deleted}")


def cleanup_transcript_duplicates():
    """Remove duplicate transcript documents keeping only the most recent."""
    print("\n" + "="*70)
    print("CLEANING DUPLICATES FROM TRANSCRIPTS")
    print("="*70)
    
    # Find duplicates: same video_id + segment_index
    pipeline = [
        {
            "$group": {
                "_id": {"video_id": "$video_id", "segment_index": "$segment_index"},
                "count": {"$sum": 1},
                "ids": {"$push": "$_id"},
                "created_at": {"$push": "$created_at"}
            }
        },
        {"$match": {"count": {"$gt": 1}}}
    ]
    
    duplicates = list(transcripts_col.aggregate(pipeline))
    
    if not duplicates:
        print("✅ No duplicates found")
        return
    
    total_deleted = 0
    for dup in duplicates:
        ids = dup["ids"]
        created_ats = dup["created_at"]
        
        # Keep the most recent
        keep_idx = created_ats.index(max(created_ats))
        ids_to_delete = ids[:keep_idx] + ids[keep_idx+1:]
        
        result = transcripts_col.delete_many({"_id": {"$in": ids_to_delete}})
        total_deleted += result.deleted_count
    
    print(f"✅ Total duplicate documents deleted: {total_deleted}")


def get_collection_stats():
    """Print statistics about current collections."""
    print("\n" + "="*70)
    print("COLLECTION STATISTICS")
    print("="*70)
    
    frames_count = frames_col.count_documents({})
    transcripts_count = transcripts_col.count_documents({})
    
    print(f"\nFrames Collection:")
    print(f"  Total documents: {frames_count}")
    
    if frames_count > 0:
        # Sample a document to check structure
        sample = frames_col.find_one()
        has_embedding = "embedding" in sample
        embedding_dim = len(sample.get("embedding", [])) if has_embedding else 0
        
        print(f"  Has embeddings: {has_embedding}")
        if has_embedding:
            print(f"  Embedding dimension: {embedding_dim}")
    
    print(f"\nTranscripts Collection:")
    print(f"  Total documents: {transcripts_count}")
    
    if transcripts_count > 0:
        sample = transcripts_col.find_one()
        has_embedding = "embedding" in sample
        embedding_dim = len(sample.get("embedding", [])) if has_embedding else 0
        
        print(f"  Has embeddings: {has_embedding}")
        if has_embedding:
            print(f"  Embedding dimension: {embedding_dim}")


def main():
    print("\n" + "="*70)
    print("CrimeVision-QA MongoDB Cleanup")
    print(f"Database: {DB_NAME}")
    print("="*70)
    
    # Show before stats
    get_collection_stats()
    
    # Clean up
    cleanup_zero_vectors()
    cleanup_duplicates()
    cleanup_transcript_duplicates()
    
    # Show after stats
    print("\n")
    get_collection_stats()
    
    print("\n" + "="*70)
    print("✅ Cleanup complete!")
    print("="*70)
    print("\nNext steps:")
    print("1. Run: python llm/process_frames.py --frames-dir frames/VideoName/ --video-id VideoName --category Category")
    print("2. Test: python debug_embeddings.py --video-id VideoName --query 'test query'")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
