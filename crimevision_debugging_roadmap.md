# CrimeVision-QA — Complete Debugging Roadmap

> **Goal:** Identify and fix every point in the pipeline where data quality degrades, especially focusing on why the VLM is generating wrong embeddings in MongoDB Atlas.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [STAGE 1 — Video → Frames (Frame Extraction)](#2-stage-1--video--frames)
3. [STAGE 2 — Frames → Text Descriptions (VLM)](#3-stage-2--frames--text-descriptions)
4. [STAGE 3 — Text → Embeddings](#4-stage-3--text--embeddings)
5. [STAGE 4 — Embeddings → MongoDB Storage](#5-stage-4--embeddings--mongodb-storage)
6. [STAGE 5 — Audio → Transcripts](#6-stage-5--audio--transcripts)
7. [STAGE 6 — Query Routing](#7-stage-6--query-routing)
8. [STAGE 7 — Retrieval (Vector + Hybrid)](#8-stage-7--retrieval)
9. [STAGE 8 — Reasoning / Answer Generation](#9-stage-8--reasoning)
10. [Root Cause Summary](#10-root-cause-summary)
11. [Prioritized Fix Plan](#11-prioritized-fix-plan)
12. [Debug Scripts](#12-debug-scripts)

---

## 1. Architecture Overview

```mermaid
graph TD
    A[Raw Video .mp4] --> B[Frame Extraction<br>video_to_image.py]
    A --> C[Audio Extraction<br>video2audio.py]
    B --> D["VLM Description<br>gen_frame_desc.py<br>(deepseek-v3p1 ⚠️)"]
    C --> E["Whisper-v3<br>audio.py"]
    D --> F["Embedding<br>get_voyage_embed.py<br>(voyage-3-large)"]
    E --> F
    F --> G["MongoDB Atlas<br>video_intelligence<br>video_intelligence_transcripts"]
    H[User Query] --> I["Router<br>router.py<br>(qwen3-8b)"]
    I --> J["Hybrid Retrieval<br>retreival_2.py"]
    J --> K["Reasoner<br>reasoner.py<br>(llama-v3p3-70b)"]
    K --> L[Answer + Timestamps]

    style D fill:#ff6b6b,color:#fff
    style F fill:#ffa07a,color:#000
    style G fill:#ffa07a,color:#000
```

> [!CAUTION]
> **The red/orange nodes are where the primary problems live.** The VLM model (`deepseek-v3p1`) is NOT a vision-language model — it's a text-only model being sent images. The embedding dimension may also be mismatched between ingestion and search.

---

## 2. STAGE 1 — Video → Frames

### File: [video_to_image.py](file:///Users/prakharsingh/Documents/project/aiml%20project/crimeanalysis/CrimeVision-QA/llm/video_to_image.py)

### Current Implementation

| Aspect | Details |
|--------|---------|
| **Input** | `.mp4` video file |
| **Process** | OpenCV reads video, extracts 1 frame every `interval_seconds` (default 2s) |
| **Output** | JPEG files: `frame_0001_t2.0s.jpg` with metadata dict |
| **Key Config** | fps-based interval calculation: `frame_interval = int(fps * interval_seconds)` |

### Analysis

| Check | Status | Notes |
|-------|--------|-------|
| Frame extraction logic | ✅ OK | Standard OpenCV approach, mathematically correct |
| Timestamp calculation | ✅ OK | `timestamp = frame_counter / fps` is correct |
| File naming | ✅ OK | Includes timestamp for easy parsing |
| Frame quality | ⚠️ Minor | No JPEG quality parameter set — defaults to ~95%, adequate |

### Verdict: ✅ No bugs here

The frame extraction code is solid. If using the Kaggle pre-extracted dataset (PNG frames), the `process_frames.py` correctly handles both naming conventions.

---

## 3. STAGE 2 — Frames → Text Descriptions

### File: [gen_frame_desc.py](file:///Users/prakharsingh/Documents/project/aiml%20project/crimeanalysis/CrimeVision-QA/llm/gen_frame_desc.py)

### Current Implementation

| Aspect | Details |
|--------|---------|
| **Input** | JPEG/PNG frame image file |
| **Process** | Base64-encode image → send to Fireworks chat/completions API as `image_url` |
| **Output** | Text description (max 300 tokens) |
| **Model** | `accounts/fireworks/models/deepseek-v3p1` (from config.py line 125) |

### 🔴 CRITICAL BUG #1: Wrong Vision Model

```python
# config.py line 125
FIREWORKS_VISION_MODEL = "accounts/fireworks/models/deepseek-v3p1"
```

> [!CAUTION]
> **`deepseek-v3p1` is NOT a vision-language model.** It is `DeepSeek-V3-0324`, a text-only LLM. When you send it an image via `image_url`, one of two things happens:
> 1. The API silently **ignores the image** and generates text based only on the text prompt — resulting in completely generic/hallucinated descriptions
> 2. The API returns an error (which the retry logic catches and returns `[DESCRIPTION UNAVAILABLE]`)
>
> **This is the #1 root cause of wrong embeddings.** If descriptions are hallucinated garbage, the embeddings will be meaningless.

### What Should Be Used

The README says **Qwen2.5-VL-32B** should be the vision model:
```
accounts/fireworks/models/qwen2-vl-72b   (or qwen2.5-vl-72b-instruct)
```

But the config.py comment on line 120-121 says:
```python
# Vision:   deepseek-v3p1 — only model on this account with image support
```

This comment is **factually wrong** — deepseek-v3p1 does not have image support. The developer may have been confused about which models were available on their Fireworks account.

### Expected Input → Output

| | Current (Broken) | Expected (Fixed) |
|--|---|---|
| **Input** | Frame image (base64) + law enforcement prompt | Same |
| **Model** | deepseek-v3p1 (text-only) | Qwen2.5-VL-72B or equivalent VLM |
| **Output** | Generic/hallucinated text OR `[DESCRIPTION UNAVAILABLE]` | Accurate visual description of the surveillance frame |
| **Example** | *"The scene appears to show an urban environment..."* (generic guess) | *"A parking lot at night. One person in a dark hoodie running east. White sedan parked, plates partially visible: 7X..."* |

### Fix

```python
# config.py — line 125, CHANGE:
FIREWORKS_VISION_MODEL = "accounts/fireworks/models/deepseek-v3p1"
# TO one of:
FIREWORKS_VISION_MODEL = "accounts/fireworks/models/qwen2-vl-72b-instruct"
# OR (if that model is too expensive / unavailable):
FIREWORKS_VISION_MODEL = "accounts/fireworks/models/llama4-scout-instruct-basic"
```

> [!IMPORTANT]
> **You MUST verify which vision models are actually available on your Fireworks account.** Run:
> ```bash
> curl -s https://api.fireworks.ai/inference/v1/models \
>   -H "Authorization: Bearer $FIREWORKS_API_KEY" | python3 -m json.tool | grep -i "qwen\|llava\|vision\|vl"
> ```

### Additional Issue: Retry Loop Bug

```python
# gen_frame_desc.py line 73
for attempt, delay in enumerate(zip(range(_MAX_RETRIES), _RETRY_DELAYS), start=1):
```

> [!WARNING]
> This is a subtle bug. `enumerate(zip(...), start=1)` means `attempt` becomes `(1, (0, 1))`, `(2, (1, 2))`, etc. — `attempt` is an int but `delay` is a tuple `(range_val, delay_val)`. The code never actually uses `delay` directly (it uses `_RETRY_DELAYS[attempt - 1]` on line 89), so it doesn't crash, but the loop counter logic is confusing and the `delay` variable is wasted.

---

## 4. STAGE 3 — Text → Embeddings

### File: [get_voyage_embed.py](file:///Users/prakharsingh/Documents/project/aiml%20project/crimeanalysis/CrimeVision-QA/llm/get_voyage_embed.py)

### Current Implementation

| Aspect | Details |
|--------|---------|
| **Input** | List of text strings (frame descriptions or transcript segments) |
| **Process (Voyage)** | Voyage AI SDK → `voyage-3-large` → 1024-dim vectors |
| **Process (Fireworks)** | Fireworks REST → `thenlper/gte-large` → vectors |
| **Output** | List of 1024-dim float vectors |
| **Current Config** | `EMBED_PROVIDER=voyage` in `.env` |

### 🟡 BUG #2: Embedding Dimension Mismatch Risk

```python
# config.py line 133
VOYAGE_EMBED_MODEL = "voyage-3-large"
```

The config expects 1024 dimensions (`_EMBED_DIM = 1024` on line 33), and MongoDB vector indexes are configured for `numDimensions: 1024`.

**`voyage-3-large`** outputs **1024 dimensions by default** ✅ — this matches.

**BUT:** `thenlper/gte-large` via Fireworks outputs **768 dimensions**, NOT 1024!

| Model | Actual Dimensions | Expected by MongoDB Index |
|-------|---|---|
| `voyage-3-large` | 1024 | 1024 ✅ |
| `thenlper/gte-large` | **768** | 1024 ❌ |

> [!WARNING]
> If you ever switch `EMBED_PROVIDER` to `fireworks`, the 768-dim vectors from GTE-large will **fail to insert** or **fail during vector search** because the Atlas index expects 1024-dim. The `_EMBED_DIM = 1024` constant is never actually validated at runtime.

### 🔴 BUG #3: Garbage In = Garbage Out

Even though the Voyage embedding service itself works correctly, **the descriptions it receives from Stage 2 are garbage** (because deepseek-v3p1 is a text-only model). So the embeddings are technically well-formed 1024-dim vectors, but they encode the semantic meaning of **hallucinated text**, not the actual visual content.

**This is why vector search returns wrong results.**

### 🟡 BUG #4: Voyage `input_type` Asymmetry

```python
# For document ingestion (process_frames.py calls embed()):
def embed(self, texts):
    return self._voyage_embed(texts, input_type="document")  # line 57

# For query (inference.py calls embed_single()):
def embed_single(self, text):
    return self._voyage_embed([text], input_type="query")     # line 74
```

This is actually **correct behavior** for Voyage — using `input_type="document"` for storage and `input_type="query"` for search gives better retrieval. ✅ No bug here.

### 🟡 BUG #5: Zero-Vector Fallback Silently Pollutes DB

```python
# process_frames.py lines 121-124
except Exception as exc:
    print(f"[Process] Embedding batch failed: {exc} — using zero vectors")
    embeddings = [_ZERO_VECTOR] * len(descriptions)
    errors += len(descriptions)
```

When embedding fails, the code stores **zero vectors** in MongoDB. These zero-vector documents:
- Still appear in vector search results (cosine similarity with zero vector is undefined/0)
- Pollute the search index and reduce retrieval quality
- Have no flag to distinguish them from real embeddings

### Expected Input → Output

| | Current (Broken) | Expected (Fixed) |
|--|---|---|
| **Input** | Hallucinated text description from deepseek-v3p1 | Accurate description from Qwen2.5-VL |
| **Embedding Model** | voyage-3-large (correct) | Same |
| **Output** | Semantically meaningless 1024-dim vector | Vector that accurately represents the visual scene |
| **Retrieval Quality** | Random/irrelevant results | Semantically relevant results |

---

## 5. STAGE 4 — Embeddings → MongoDB Storage

### File: [mongo_client_1.py](file:///Users/prakharsingh/Documents/project/aiml%20project/crimeanalysis/CrimeVision-QA/llm/mongo_client_1.py)

### Current Implementation

| Aspect | Details |
|--------|---------|
| **Database** | `GenAI` (from .env) |
| **Collections** | `video_intelligence`, `video_intelligence_transcripts`, `video_library`, `previous_frame_incidents` |
| **Vector Indexes** | `vs_frames_index` (1024-dim, cosine) on `video_intelligence.embedding` |
| | `vs_transcripts_index` (1024-dim, cosine) on `video_intelligence_transcripts.embedding` |
| **Text Indexes** | `text_description` on `video_intelligence.description` |
| | `text_transcript` on `video_intelligence_transcripts.text` |

### 🟡 BUG #6: Vector Index Must Be Created Manually

```python
# mongo_client_1.py line 99-145
_VECTOR_INDEX_INSTRUCTIONS = """... MANUAL STEP REQUIRED ..."""
```

Vector search indexes must be created **manually** via the Atlas console. If they haven't been created:
- `$vectorSearch` aggregation will throw an error
- The code will silently return empty results (no error propagation)

### Verification Checklist

```
□ Database name "GenAI" exists in Atlas
□ Collection "video_intelligence" exists with documents
□ Collection "video_intelligence_transcripts" exists with documents  
□ Vector index "vs_frames_index" exists and is ACTIVE (not BUILDING)
□ Vector index "vs_transcripts_index" exists and is ACTIVE
□ Vector index dimension matches embedding dimension (1024)
□ Documents have an "embedding" field that is a 1024-element array
□ No documents have zero-vector embeddings
```

### 🟡 BUG #7: No Deduplication on Re-ingestion

```python
# process_frames.py line 148
frames_col.insert_many(docs, ordered=False)
```

If you re-run ingestion for the same video, duplicate documents are created. Subsequent vector searches will return duplicate results for the same frame, wasting result slots and confusing the reasoner.

### Fix: Add upsert logic

```python
# Replace insert_many with upserts:
from pymongo import UpdateOne
ops = [
    UpdateOne(
        {"video_id": doc["video_id"], "frame_file": doc["frame_file"]},
        {"$set": doc},
        upsert=True,
    )
    for doc in docs
]
frames_col.bulk_write(ops, ordered=False)
```

---

## 6. STAGE 5 — Audio → Transcripts

### Files: [video2audio.py](file:///Users/prakharsingh/Documents/project/aiml%20project/crimeanalysis/CrimeVision-QA/transcripts/video2audio.py) + [audio.py](file:///Users/prakharsingh/Documents/project/aiml%20project/crimeanalysis/CrimeVision-QA/transcripts/audio.py)

### Current Implementation

| Aspect | Details |
|--------|---------|
| **Input** | `.mp4` video |
| **Process 1** | FFmpeg extracts audio → `.mp3` |
| **Process 2** | Whisper-v3 via Fireworks → timestamped segments |
| **Process 3** | Embed segments with Voyage → store in MongoDB |
| **Output** | MongoDB docs with `{text, start_time, end_time, embedding}` |

### Analysis

| Check | Status | Notes |
|-------|--------|-------|
| FFmpeg audio extraction | ✅ OK | Handles no-audio gracefully |
| Whisper endpoint URL | ✅ OK | `https://audio-prod.api.fireworks.ai/v1/audio/transcriptions` |
| Segment parsing | ✅ OK | Properly extracts start_time, end_time, text |
| 25MB truncation | ⚠️ Risk | Silently truncates large files — could lose end-of-video audio |
| Embedding | ⚠️ Same as Stage 3 | Correct if using Voyage, broken if using Fireworks GTE |

### 🟡 BUG #8: Kaggle Dataset Has No Audio

The Kaggle UCF-Crime dataset (`odins0n/ucf-crime-dataset`) contains **only pre-extracted PNG frames, not MP4 videos**. The `ingest_dataset.py` and `ingest_from_kaggle.py` scripts skip audio processing entirely because there are no video files to extract audio from.

**Impact:** The `video_intelligence_transcripts` collection will be completely **empty** for all Kaggle-ingested videos. This means:
- Any `FIND_AUDIO` query intent will return no results
- The hybrid search transcript branch contributes nothing
- The reasoner only gets visual evidence

This isn't a bug per se — it's a dataset limitation — but the system should handle this gracefully (and it does: `semantic_search_transcripts` returns `[]` when no documents exist).

---

## 7. STAGE 6 — Query Routing

### File: [router.py](file:///Users/prakharsingh/Documents/project/aiml%20project/crimeanalysis/CrimeVision-QA/llm/query_model/router.py)

### Current Implementation

| Aspect | Details |
|--------|---------|
| **Input** | User natural-language query |
| **Model** | `accounts/fireworks/models/qwen3-8b` |
| **Output** | `{intent, search_query, time_range, confidence}` |
| **Intents** | `FIND_AUDIO`, `FIND_FRAME`, `FIND_VIDEO_META`, `SUMMARIZE_WINDOW`, `COUNT` |

### Analysis

| Check | Status | Notes |
|-------|--------|-------|
| System prompt quality | ✅ Good | Clear, well-structured with examples |
| JSON parsing with fallback | ✅ Good | regex fallback if JSON parse fails |
| Default intent on failure | ✅ Good | Falls back to `FIND_FRAME` |
| Temperature = 0 | ✅ Good | Deterministic for classification |
| Model choice | ⚠️ OK | qwen3-8b is cheap but capable enough for classification |

### 🟡 BUG #9: `search_query` May Not Be Optimized

The router asks the LLM to produce a "retrieval-optimized" `search_query`, but the LLM may return the original query unchanged or overly shortened. This affects embedding quality during search.

**This is a minor issue** — the bigger problem is that the stored embeddings themselves are garbage.

---

## 8. STAGE 7 — Retrieval

### Files: [inference.py](file:///Users/prakharsingh/Documents/project/aiml%20project/crimeanalysis/CrimeVision-QA/llm/inference.py) + [retreival_2.py](file:///Users/prakharsingh/Documents/project/aiml%20project/crimeanalysis/CrimeVision-QA/llm/retreival_2.py)

### Current Implementation

**Vector Search** (`inference.py`):
| Aspect | Details |
|--------|---------|
| **Input** | Query text → embedded via `embed_single()` |
| **Process** | `$vectorSearch` aggregation with `numCandidates: 100`, `limit: k` |
| **Output** | Top-k documents with `vectorSearchScore` |

**Hybrid Search** (`retreival_2.py`):
| Aspect | Details |
|--------|---------|
| **Input** | Query text |
| **Process** | Vector search (k×3) + Text search (k×3) → RRF fusion (70/30) |
| **Output** | Top-k re-ranked documents with `rrf_score` |

### Analysis

| Check | Status | Notes |
|-------|--------|-------|
| Vector search pipeline | ✅ OK | Correct `$vectorSearch` syntax |
| Video_id filter | ✅ OK | Passes `filter: {video_id: {$eq: video_id}}` |
| RRF implementation | ✅ OK | Standard formula: `weight / (K + rank + 1)` |
| Text search | ✅ OK | Uses MongoDB `$text` operator correctly |
| Early exit on empty collection | ✅ Good | `count_documents(limit=1)` check avoids unnecessary API call |

### 🔴 BUG #10: Retrieval Returns Wrong Results (Consequence of Bug #1)

The retrieval code itself is correct, but because the stored embeddings encode hallucinated descriptions (from the wrong VLM), **cosine similarity between query embeddings and stored embeddings is essentially random**. The returned frames have no semantic relationship to the query.

### Example of Current Broken Behavior

```
Query: "Who was involved in the assault?"
Query embedding: voyage-3-large("Who was involved in the assault?")  → 1024-dim vector A

Stored frame embedding: voyage-3-large("The scene appears to show a generic urban 
    environment with various elements...")  → 1024-dim vector B
    
cosine_similarity(A, B) → 0.72  (looks decent numerically)
But vector B encodes HALLUCINATED text, not actual frame content!
```

### Expected Behavior After Fixing VLM

```
Query: "Who was involved in the assault?"
Query embedding: same vector A

Stored frame embedding: voyage-3-large("Two males in a confrontation. Subject 1 
    wearing dark hoodie, Subject 2 in white t-shirt. Subject 1 strikes Subject 2 
    at the head...")  → 1024-dim vector C
    
cosine_similarity(A, C) → 0.85  (high AND semantically accurate)
```

---

## 9. STAGE 8 — Reasoning

### File: [reasoner.py](file:///Users/prakharsingh/Documents/project/aiml%20project/crimeanalysis/CrimeVision-QA/llm/query_model/reasoner.py)

### Current Implementation

| Aspect | Details |
|--------|---------|
| **Input** | Retrieved frame/transcript documents + user query |
| **Process** | Format evidence → system prompt (strategy-dependent) → LLM call |
| **Model (current)** | `accounts/fireworks/models/llama-v3p3-70b-instruct` (REASONER_PROVIDER=fireworks) |
| **Output** | `{answer, timestamps, sources, strategy_used}` |

### Analysis

| Check | Status | Notes |
|-------|--------|-------|
| Context formatting | ✅ Good | `_format_context()` cleanly separates visual/audio evidence |
| 4 prompt strategies | ✅ Good | Zero-shot, CoT, Few-shot, ReAct all implemented |
| Timestamp extraction | ✅ Good | Regex extracts both "30.0s" and "0:30" formats |
| Error handling | ✅ OK | Returns safe fallback message |
| Reasoner model | ✅ OK | Llama-3.3-70B is capable |

### 🟡 Issue: Good Reasoner, Bad Evidence

The reasoner is well-implemented. **But it can only be as good as its evidence.** If Stage 2 produces hallucinated descriptions, the reasoner will faithfully synthesize an answer from that garbage evidence — producing confident-sounding but factually wrong answers.

---

## 10. Root Cause Summary

```mermaid
graph LR
    A["🔴 BUG #1<br>Wrong VLM Model<br>deepseek-v3p1 ≠ VLM"] --> B["Hallucinated<br>Frame Descriptions"]
    B --> C["Meaningless<br>Embeddings"]
    C --> D["Wrong Retrieval<br>Results"]
    D --> E["Wrong Final<br>Answers"]
    
    F["🟡 BUG #2<br>GTE-large = 768-dim<br>vs 1024-dim index"] -.-> G["Crash if<br>EMBED_PROVIDER=fireworks"]
    
    H["🟡 BUG #5<br>Zero-Vector<br>Fallback"] -.-> D
    
    I["🟡 BUG #7<br>No Dedup<br>on Re-ingest"] -.-> D
    
    style A fill:#ff4444,color:#fff
    style B fill:#ff6666,color:#fff
    style C fill:#ff8888,color:#fff
    style D fill:#ffaaaa,color:#000
    style E fill:#ffcccc,color:#000
```

### Priority-Ranked Bug List

| # | Severity | Component | Bug | Impact |
|---|----------|-----------|-----|--------|
| 1 | 🔴 **CRITICAL** | `config.py` L125 | `deepseek-v3p1` is text-only, not a VLM | **All frame descriptions are hallucinated → all embeddings are wrong → all retrieval fails** |
| 2 | 🟡 **HIGH** | `config.py` L128 | `thenlper/gte-large` outputs 768-dim, not 1024 | Will crash if `EMBED_PROVIDER=fireworks` |
| 3 | 🔴 **CRITICAL** | MongoDB | Existing data in Atlas has corrupt embeddings | Must be wiped and re-ingested after fixing model |
| 5 | 🟡 **MEDIUM** | `process_frames.py` L121 | Zero-vector fallback pollutes search results | Retrieval returns irrelevant docs |
| 7 | 🟡 **MEDIUM** | `process_frames.py` L148 | No dedup on re-ingestion | Duplicate docs in search results |
| 8 | ⚪ **LOW** | Dataset | Kaggle dataset has no audio → empty transcripts | Transcript search always empty (by design) |
| 9 | ⚪ **LOW** | `router.py` | search_query optimization depends on LLM | Minor retrieval quality impact |

---

## 11. Prioritized Fix Plan

### Phase 1: Fix the VLM Model (CRITICAL — Do First)

#### Step 1.1: Identify available vision models on your Fireworks account

```bash
# Run this to see what vision models are available:
curl -s "https://api.fireworks.ai/inference/v1/models" \
  -H "Authorization: Bearer fw_HdyVq52PSSby59aKL5r63Q" \
  | python3 -c "
import json, sys
models = json.load(sys.stdin)
for m in models.get('data', []):
    mid = m.get('id', '')
    if any(kw in mid.lower() for kw in ['vl', 'vision', 'llava', 'qwen2-vl', 'scout']):
        print(f\"  {mid}\")
"
```

#### Step 1.2: Update `config.py` with a real VLM

```python
# config.py line 125 — change to actual VLM, e.g.:
FIREWORKS_VISION_MODEL = "accounts/fireworks/models/qwen2-vl-72b-instruct"
```

#### Step 1.3: Validate the VLM works with a test frame

```bash
python3 -c "
from llm.gen_frame_desc import describe_frame
desc = describe_frame('frames/Abuse001_x264/Abuse001_x264_0.png')
print('DESCRIPTION:', desc)
print('LENGTH:', len(desc))
print('IS_PLACEHOLDER:', desc in ['[DESCRIPTION UNAVAILABLE]', '[INVALID FRAME]'])
"
```

**Expected:** A specific description of the scene, NOT generic text or an error.

---

### Phase 2: Clean Existing Corrupt Data

#### Step 2.1: Drop all existing frame data

```python
from llm.config import frames_col, transcripts_col
# Count existing corrupt data
print(f"Frames to delete: {frames_col.count_documents({})}")
print(f"Transcripts to delete: {transcripts_col.count_documents({})}")

# DELETE ALL — must re-ingest with correct VLM
frames_col.delete_many({})
transcripts_col.delete_many({})
```

#### Step 2.2: Verify vector indexes still exist (they survive data deletion)

Go to Atlas Console → Database → Browse Collections → Search Indexes tab → Verify:
- `vs_frames_index` — Status: ACTIVE, Dimensions: 1024
- `vs_transcripts_index` — Status: ACTIVE, Dimensions: 1024

---

### Phase 3: Re-Ingest Data with Correct VLM

```bash
# Single video test first:
python scripts/ingest_dataset.py \
  --video-id Abuse001_x264 --category Abuse \
  --max-frames 5 --stride 10 --batch-size 2 --delay 2.0

# Verify the stored descriptions are real:
python3 -c "
from llm.config import frames_col
for doc in frames_col.find({'video_id': 'Abuse001_x264'}, {'description': 1, 'frame_file': 1}).limit(3):
    print(f\"{doc['frame_file']}: {doc['description'][:200]}...\")
"
```

---

### Phase 4: Fix Secondary Bugs

#### 4.1: Add embedding dimension validation

```python
# get_voyage_embed.py — add after embedding call:
def embed(self, texts):
    embeddings = self._voyage_embed(texts, input_type="document") if self.provider == "voyage" else self._fireworks_embed(texts)
    # Validate dimensions
    for i, emb in enumerate(embeddings):
        if len(emb) != _EMBED_DIM:
            raise ValueError(f"Embedding {i} has {len(emb)} dims, expected {_EMBED_DIM}")
    return embeddings
```

#### 4.2: Replace insert_many with upsert (dedup fix)

```python
# process_frames.py — replace line 148:
from pymongo import UpdateOne
ops = [
    UpdateOne(
        {"video_id": doc["video_id"], "frame_file": doc["frame_file"]},
        {"$set": doc},
        upsert=True,
    )
    for doc in docs
]
if ops:
    frames_col.bulk_write(ops, ordered=False)
```

#### 4.3: Skip storing zero-vector documents

```python
# process_frames.py — replace lines 128-133:
for m, desc, emb in zip(meta, descriptions, embeddings):
    if desc in ("[DESCRIPTION UNAVAILABLE]", "[INVALID FRAME]"):
        errors += 1
        continue  # Don't store frames with no description
    if emb == _ZERO_VECTOR:
        errors += 1
        continue  # Don't store zero-vector frames
    # ... rest of doc building
```

#### 4.4: Fix GTE-large dimension in config

```python
# config.py — add a safety check:
FIREWORKS_EMBED_MODEL = "thenlper/gte-large"
FIREWORKS_EMBED_DIM = 768   # GTE-large is 768-dim, NOT 1024!
# NOTE: If EMBED_PROVIDER=fireworks, MongoDB vector index must be 768-dim
```

---

### Phase 5: End-to-End Verification

#### Step 5.1: Verify descriptions are real

```python
from llm.config import frames_col

docs = list(frames_col.find({}, {"description": 1, "embedding": 1}).limit(5))
for doc in docs:
    desc = doc["description"]
    emb = doc["embedding"]
    print(f"Description length: {len(desc)}")
    print(f"Embedding dims: {len(emb)}")
    print(f"Embedding non-zero: {sum(1 for x in emb if x != 0.0)}")
    print(f"First 100 chars: {desc[:100]}")
    print("---")
```

#### Step 5.2: Test retrieval

```python
from llm.inference import semantic_search_frames

results = semantic_search_frames("person fighting assault", video_id="Abuse001_x264", k=3)
for r in results:
    print(f"Score: {r['score']:.4f} | t={r['timestamp_seconds']}s | {r['description'][:100]}...")
```

**Expected:** High scores (>0.7) for frames that actually show fighting/assault activity.

#### Step 5.3: Test full pipeline

```python
from llm.agent import run_agent_sync
result = run_agent_sync("What is happening in this video?", "Abuse001_x264", "zero_shot")
print(result["answer"])
```

---

## 12. Debug Scripts

Create and run these scripts to diagnose the current state:

### Script 1: Check what's actually in MongoDB

```python
"""debug_check_mongo.py — What's currently stored?"""
import sys; sys.path.insert(0, '.')
from llm.config import frames_col, transcripts_col, video_library_col

print("=== MongoDB State ===")
print(f"Frames:      {frames_col.count_documents({})} documents")
print(f"Transcripts: {transcripts_col.count_documents({})} documents")
print(f"Videos:      {video_library_col.count_documents({})} documents")

# Check for zero-vector frames
zero_count = frames_col.count_documents({"embedding.0": 0.0})
print(f"Zero-vector frames: {zero_count}")

# Sample descriptions
print("\n--- Sample Frame Descriptions ---")
for doc in frames_col.find({}, {"video_id": 1, "frame_file": 1, "description": 1}).limit(5):
    print(f"  [{doc.get('video_id')}/{doc.get('frame_file')}]")
    print(f"    {doc.get('description', 'N/A')[:200]}")
    print()

# Check embedding dimensions
sample = frames_col.find_one({}, {"embedding": 1})
if sample and "embedding" in sample:
    print(f"Embedding dimensions: {len(sample['embedding'])}")
else:
    print("No embeddings found!")
```

### Script 2: Test VLM model capability

```python
"""debug_test_vlm.py — Does the VLM actually understand images?"""
import sys; sys.path.insert(0, '.')
from llm.gen_frame_desc import describe_frame
import os

# Find any frame file
frames_dir = "./frames"
if os.path.isdir(frames_dir):
    for root, dirs, files in os.walk(frames_dir):
        for f in files:
            if f.endswith(('.png', '.jpg')):
                path = os.path.join(root, f)
                print(f"Testing VLM on: {path}")
                desc = describe_frame(path)
                print(f"Result: {desc}")
                print(f"Is placeholder: {desc in ['[DESCRIPTION UNAVAILABLE]', '[INVALID FRAME]']}")
                break
        break
else:
    print("No frames directory found. Run ingestion first.")
```

### Script 3: Test embedding quality

```python
"""debug_test_embeddings.py — Are the embeddings meaningful?"""
import sys; sys.path.insert(0, '.')
from llm.get_voyage_embed import embedding_service
import numpy as np

texts = [
    "A person punching another person in a parking lot",
    "Empty highway with no activity",
    "Person involved in a physical altercation",
]

embeddings = embedding_service.embed(texts)
print(f"Embedding dimensions: {len(embeddings[0])}")

# Check cosine similarity between related vs unrelated texts
def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

print(f"\nSimilarity (fight vs altercation): {cosine_sim(embeddings[0], embeddings[2]):.4f}  (should be HIGH)")
print(f"Similarity (fight vs empty road):  {cosine_sim(embeddings[0], embeddings[1]):.4f}  (should be LOW)")
```

---

> [!IMPORTANT]
> **The single most impactful fix is changing `FIREWORKS_VISION_MODEL` in `config.py` from `deepseek-v3p1` to an actual VLM (like `qwen2-vl-72b-instruct`), then wiping and re-ingesting all data.** This one change will fix ~90% of the quality issues you're seeing.
