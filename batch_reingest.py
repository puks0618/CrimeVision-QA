#!/usr/bin/env python3
"""
Batch re-ingest all videos with improved prompt
"""
import subprocess
import sys
from pathlib import Path

# Video configurations (video_dir, video_id, category)
VIDEOS = [
    ("Arrest002_x264", "Arrest002_x264", "Arrest"),
    ("Arson001_x264", "Arson001_x264", "Arson"),
    ("Assault001_x264", "Assault001_x264", "Assault"),
    ("Burglary001_x264", "Burglary001_x264", "Burglary"),
    ("Explosion001_x264", "Explosion001_x264", "Explosion"),
    ("Fighting002_x264", "Fighting002_x264", "Fighting"),
    ("Normal_Videos001_x264", "Normal_Videos001_x264", "Normal"),
    ("RoadAccidents003_x264", "RoadAccidents003_x264", "RoadAccident"),
    ("Robbery001_x264", "Robbery001_x264", "Robbery"),
    ("Shooting001_x264", "Shooting001_x264", "Shooting"),
    ("Shoplifting003_x264", "Shoplifting003_x264", "Shoplifting"),
    ("Stealing002_x264", "Stealing002_x264", "Stealing"),
    ("Vandalism001_x264", "Vandalism001_x264", "Vandalism"),
]

BASE_DIR = Path.cwd()
FRAMES_DIR = BASE_DIR / "frames"

def check_video_dir(video_dir):
    """Check if video directory exists"""
    full_path = FRAMES_DIR / video_dir
    if not full_path.exists():
        return False
    frame_count = len(list(full_path.glob("*.png")))
    return frame_count > 0

def reingest_video(video_dir, video_id, category):
    """Re-ingest a single video"""
    frames_path = FRAMES_DIR / video_dir
    
    if not frames_path.exists():
        print(f"❌ {video_id}: Directory not found - {frames_path}")
        return False
    
    frame_count = len(list(frames_path.glob("*.png")))
    if frame_count == 0:
        print(f"❌ {video_id}: No frames found")
        return False
    
    print(f"\n{'='*70}")
    print(f"📹 Processing: {video_id} ({frame_count} frames)")
    print(f"{'='*70}")
    
    cmd = [
        "python3",
        "llm/process_frames.py",
        "--frames-dir", str(frames_path),
        "--video-id", video_id,
        "--category", category,
    ]
    
    try:
        result = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True, timeout=1800)
        
        if result.returncode == 0:
            print(f"✅ {video_id}: Successfully re-ingested")
            # Print last few lines of output
            lines = result.stderr.split('\n') if result.stderr else result.stdout.split('\n')
            for line in lines[-5:]:
                if line.strip():
                    print(f"   {line}")
            return True
        else:
            print(f"❌ {video_id}: Re-ingestion failed")
            print(f"Error: {result.stderr[-500:] if result.stderr else result.stdout[-500:]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏱️  {video_id}: Timeout (> 30 minutes)")
        return False
    except Exception as e:
        print(f"❌ {video_id}: Exception - {e}")
        return False

def main():
    print("\n" + "="*70)
    print("🚀 BATCH RE-INGESTION - IMPROVED PROMPT")
    print("="*70)
    print(f"Total videos to process: {len(VIDEOS)}")
    
    success_count = 0
    failed_videos = []
    
    for video_dir, video_id, category in VIDEOS:
        if check_video_dir(video_dir):
            if reingest_video(video_dir, video_id, category):
                success_count += 1
            else:
                failed_videos.append(video_id)
        else:
            print(f"⏭️  {video_id}: Directory or frames not found, skipping")
    
    print("\n" + "="*70)
    print("📊 BATCH EXECUTION SUMMARY")
    print("="*70)
    print(f"✅ Successful: {success_count}/{len(VIDEOS)}")
    print(f"❌ Failed: {len(failed_videos)}/{len(VIDEOS)}")
    
    if failed_videos:
        print(f"\nFailed videos: {', '.join(failed_videos)}")
    
    print("="*70)
    return 0 if len(failed_videos) == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
