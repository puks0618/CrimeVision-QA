# CrimeVision-QA Embedding Mismatch — Complete Fix Guide

## Problem Summary

Your video context descriptions don't match actual video actions. This breaks the entire retrieval pipeline because:

```
Bad Descriptions → Meaningless Embeddings → Wrong Vector Search Results → Wrong Answers
```

## Root Causes (Priority Order)

### 🔴 CRITICAL: Vision Model Description Quality

**File**: `llm/gen_frame_desc.py` 
**Root Cause**: Frame descriptions are **inaccurate or hallucinated**

The pipeline now uses `kimi-k2p5` (correct VLM), but descriptions may still be wrong if:

1. **Vision model prompt is too generic** 
   - Current prompt at line 13-15 is generic for "law enforcement"
   - Doesn't ask for specific details about suspicious activity
   
2. **No quality validation after description generation**
   - Descriptions with low confidence still get stored
   - No filtering for generic/unhelpful descriptions

3. **Vision model may be ignoring important visual details**
   - Model might not be trained well for surveillance video
   - May miss fine-grained details (clothing, faces, vehicle details)

**IMMEDIATE FIX**: 

```python
# llm/gen_frame_desc.py — IMPROVE THE PROMPT

# OLD (too generic):
_PROMPT = (
    "Describe this surveillance video frame for law enforcement analysis. "
    "Include: people (appearance, clothing, actions, positions), vehicles "
    "(type, color, partial plates if visible), objects, setting, lighting "
    "conditions, and any visible text or signage. Be concise and factual."
)

# NEW (specific to CrimeVision):
_PROMPT = (
    "Analyze this surveillance frame for criminal activity detection. "
    "MANDATORY: Report the following with absolute precision:\n"
    "1. PEOPLE: Count, ethnicities, clothing colors/types, visible faces, "
    "   exact actions (running/standing/fighting/stealing), hand positions, "
    "   weapons visible, injuries visible\n"
    "2. VEHICLES: Type, color, visible plate digits, occupants, damage\n"
    "3. OBJECTS: Weapons, stolen items, tools, packages\n"
    "4. SETTING: Indoor/outdoor, location type (street/store/parking lot), "
    "   entry/exit points visible\n"
    "5. UNUSUAL: Any suspicious behavior, rapid movements, concealment\n"
    "Be EXTREMELY specific. Avoid generic descriptions like 'appears to show'. "
    "If uncertain about a detail, say '[UNCERTAIN]' instead of guessing."
)
```

### 🟡 HIGH PRIORITY: Embedding Dimension Mismatch

**File**: `llm/config.py` lines 100-103, `llm/get_voyage_embed.py` line 33

**Issue**: If MongoDB index expects 1024-dim but gets different dimension:
- Vector search silently fails or returns empty results
- No error is raised to alert you

**Check**:
```bash
# Verify your MongoDB index dimension matches embeddings
# MongoDB Atlas → Database → video_intelligence → Search Indexes → vs_frames_index
# Look for: "numDimensions": 1024 (or 768 if using Fireworks)
```

**Current Config**:
- `EMBED_PROVIDER`: voyage
- `VOYAGE_EMBED_MODEL`: voyage-3-large (outputs 1024-dim ✅)
- `FIREWORKS_EMBED_MODEL`: gte-large (outputs 768-dim ⚠️)

**Never switch to Fireworks embeddings unless** you update MongoDB index to 768-dim.

### 🟡 MEDIUM PRIORITY: Zero-Vector Pollution

**File**: `llm/process_frames.py` lines 127-132

**Status**: Already handled ✅
- Zero vectors are skipped (not stored)
- Failed descriptions are not stored
- This is correct

**But**: Need to clean up any old zero vectors from previous runs:

```bash
# Connect to MongoDB and run:
db.video_intelligence.deleteMany({ 
  embedding: { $eq: Array(1024).fill(0) } 
})

db.video_intelligence_transcripts.deleteMany({ 
  embedding: { $eq: Array(1024).fill(0) } 
})
```

### 🟡 MEDIUM PRIORITY: Re-ingestion Duplicates

**File**: `llm/process_frames.py` lines 138-143

**Status**: Already handled ✅
- Uses upsert logic (UpdateOne, not insert_many)
- Duplicates are automatically replaced

**But**: Clean up any old duplicates:

```bash
# Connect to MongoDB and run:
db.video_intelligence.aggregate([
  {
    $group: {
      _id: { video_id: "$video_id", frame_file: "$frame_file" },
      count: { $sum: 1 },
      ids: { $push: "$_id" }
    }
  },
  { $match: { count: { $gt: 1 } } }
])

# If duplicates exist, keep only the latest:
# (Need to delete manually or use a cleanup script)
```

## Step-by-Step Fix

### Step 1: Improve Vision Model Prompt

```bash
# 1. Edit gen_frame_desc.py
nano llm/gen_frame_desc.py

# 2. Replace _PROMPT with the improved version above (lines 13-15)

# 3. Save and close
```

### Step 2: Verify MongoDB Index Configuration

```bash
# 1. Open MongoDB Atlas console
# 2. Navigate to: Database → Collections → video_intelligence
# 3. Click "Search Indexes" tab
# 4. Check "vs_frames_index" details:
#    - Status should be "ACTIVE" (not BUILDING)
#    - numDimensions should be 1024
#    - Similarity metric should be "cosine"

# Example correct index:
{
  "name": "vs_frames_index",
  "type": "vectorSearch",
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "similarity": "cosine",
      "dimensions": 1024
    }
  ]
}
```

### Step 3: Re-ingest Video with Improved Prompt

```bash
# 1. Delete old frames for this video (optional, upsert will replace)
python -c "
from llm.config import frames_col
video_id = 'Abuse001_x264'  # Change to your video
result = frames_col.delete_many({'video_id': video_id})
print(f'Deleted {result.deleted_count} old frames')
"

# 2. Re-ingest with improved descriptions:
python llm/process_frames.py \
  --frames-dir frames/Abuse001_x264/ \
  --video-id Abuse001_x264 \
  --category Abuse

# 3. Monitor output for:
#    - Frame descriptions are now MORE DETAILED
#    - No "[DESCRIPTION UNAVAILABLE]" errors
#    - All embeddings are stored (no "using zero vectors")
```

### Step 4: Test Vector Search Quality

```bash
# Run the diagnostic script:
python debug_embeddings.py \
  --video-id Abuse001_x264 \
  --query "person running" \
  --output diagnostic_report.json

# Expected output:
# ✅ Stage 1: Vision descriptions are detailed and accurate
# ✅ Stage 2: Embedding dimension matches (1024-dim)
# ✅ Stage 3: MongoDB indexes exist and are ACTIVE
# ✅ Stage 4: Vector search returns relevant results
```

### Step 5: Evaluate Retrieval Quality

```bash
# Test retrieval with evaluation script:
python evaluation/eval_retrieval.py \
  --video-id Abuse001_x264 \
  --queries evaluation/test_queries.json

# Check: Are the top-5 results NOW relevant to the query?
```

## Validation Checklist

Before and after fix, run:

```python
from llm.config import frames_col
from llm.inference import semantic_search_frames

# Check 1: Descriptions are detailed
doc = frames_col.find_one({"video_id": "Abuse001_x264"})
print(f"Description length: {len(doc['description'])}")  # Should be 150+ chars
print(f"Description: {doc['description'][:200]}")  # Should be specific, not generic

# Check 2: Embeddings are valid
print(f"Embedding dimension: {len(doc['embedding'])}")  # Should be 1024
print(f"Embedding sample: {doc['embedding'][:5]}")  # Should be non-zero floats

# Check 3: Vector search works
results = semantic_search_frames("person fighting", video_id="Abuse001_x264", k=3)
print(f"Search returned {len(results)} results")  # Should be > 0
print(f"Top result description: {results[0]['description'][:100]}")  # Should match query
```

## If Issues Persist

### Problem: Descriptions still seem generic/inaccurate

**Solution**: Try alternative vision models

```python
# llm/config.py line 114
# Try one of these:

# Option 1: Claude Vision (if available)
FIREWORKS_VISION_MODEL = "accounts/fireworks/models/claude-3-5-sonnet"

# Option 2: GPT-4V equivalent  
FIREWORKS_VISION_MODEL = "accounts/fireworks/models/qwen2-vl-72b-instruct"

# Option 3: Open-source alternative
FIREWORKS_VISION_MODEL = "accounts/fireworks/models/llava-1.6-7b"
```

### Problem: Vector search still returns irrelevant results

**Check these in order**:

1. **MongoDB index isn't active**: Status should be "ACTIVE", not "BUILDING"
   - Wait for index to build or delete and recreate

2. **Query embedding isn't being generated**: 
   ```python
   from llm.get_voyage_embed import embedding_service
   query_vec = embedding_service.embed_single("test query")
   print(f"Query embedding dimension: {len(query_vec)}")  # Should be 1024
   ```

3. **No matching documents exist**: 
   ```python
   from llm.config import frames_col
   count = frames_col.count_documents({"video_id": "Abuse001_x264"})
   print(f"Total frames in DB: {count}")  # Should be > 0
   ```

## Summary of Changes

| Component | Before | After | Status |
|-----------|--------|-------|--------|
| Vision Model | deepseek-v3p1 (text-only) | kimi-k2p5 (VLM) | ✅ Already Updated |
| Vision Prompt | Generic | Specific + Detailed | 🔄 Need Fix |
| Embedding Dim Check | No validation | Validated at runtime | ✅ Implemented |
| Zero-Vector Handling | Silent fallback | Skip + Flag | ✅ Already Fixed |
| Re-ingestion | Duplicates | Upsert (no dups) | ✅ Already Fixed |
| MongoDB Index | Manual creation | Must verify active | 🔄 Need Check |

## Key Takeaway

The code infrastructure is mostly fixed (vision model, upsert logic, dimension checks). **The issue is now likely DESCRIPTION QUALITY** — the improved prompt above should resolve it. If not, you may need to:

1. Switch to a better vision model
2. Add temperature/quality controls to descriptions
3. Implement manual review + feedback loop for descriptions

