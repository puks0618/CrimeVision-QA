# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**CrimeVision-QA** (SafeGuard AI) — a multimodal RAG system for querying surveillance and body-camera footage via natural language. It extracts frames, describes them with a vision model, transcribes audio, embeds everything into MongoDB Atlas, and answers questions with timestamped evidence using a LangGraph ReAct agent.

## Commands

### Backend
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend && npm run dev        # http://localhost:5173
cd frontend && npm run build      # TypeScript check + Vite build
```

### Install dependencies
```bash
python -m pip install -r requirements.txt
cd frontend && npm install
```

### Video ingestion (triggered automatically via POST /api/upload, but also runnable standalone)
```bash
python batch_reingest.py          # Re-ingest all videos in ./videos/
python setup_indexes.py           # Create/verify MongoDB vector + text indexes
python cleanup_mongodb.py         # Drop all collections (destructive)
```

### Evaluation
```bash
python evaluation/eval_matrix.py                    # BLEU/ROUGE-L/BERTScore/SemScore × 4 strategies
python evaluation/eval_matrix.py --limit 1          # Smoke test (1 query)
python evaluation/eval_ragas.py --video-id <id>     # Faithfulness, relevancy, context precision
python evaluation/eval_retrieval.py --video-id <id> # Recall@K, intent accuracy
python evaluation/eval_prompts.py --video-id <id>   # Strategy latency + answer quality
python evaluation/eval_latency.py                   # End-to-end latency benchmarks
```

### Fine-tuning (GPU required)
```bash
python finetune/generate_training_data.py           # Build instruction-response pairs from MongoDB
python finetune/train_qlora.py                      # QLoRA on Llama-3.1-8B-Instruct
python finetune/eval_finetuned.py --adapter-path finetune/output/checkpoint-final --video-id <id>
```

## Environment Variables

Copy `.env` to the project root. Required:

| Variable | Purpose |
|---|---|
| `FIREWORKS_API_KEY` | Vision (kimi-k2p5), audio (Whisper-v3), routing + reasoning fallback (Llama-3.3-70B), embeddings fallback (GTE-large) |
| `MONGODB_URI` | MongoDB Atlas connection string |
| `MONGODB_DB_NAME` | Database name (default: `video_intelligence`) |
| `VOYAGE_API_KEY` | Voyage AI embeddings (`voyage-3-large`, 1024-dim). If unset, falls back to Fireworks GTE-large (768-dim) |
| `GEMINI_API_KEY` | Gemini 2.5 Flash for answer generation. If unset, falls back to Fireworks Llama-3.3-70B |

**Critical:** switching embedding providers changes vector dimension (Voyage=1024, Fireworks GTE=768). MongoDB vector indexes must be rebuilt after switching — run `cleanup_mongodb.py` then `setup_indexes.py`.

## Architecture

### End-to-End Request Flow

```
POST /api/chat  (backend/main.py)
  └─ run_agent(query, video_id, strategy)   [llm/agent.py]
       ├─ route_query()                      [llm/query_model/router.py]
       │    → classifies into 5 intents: FIND_FRAME | FIND_AUDIO |
       │      SUMMARIZE_WINDOW | COUNT | FIND_VIDEO_META
       ├─ hybrid_search_frames/transcripts() [llm/retreival_2.py]
       │    → vector search + keyword search → manual RRF (70/30 weight)
       │      (avoids $rankFusion — works on MongoDB M0 free tier)
       └─ reasoner.reason()                  [llm/query_model/reasoner.py]
            → one of 4 strategies:
              zero_shot | cot | few_shot | react
            → LLM: Gemini 2.5 Flash or Fireworks Llama-3.3-70B
```

### Video Ingestion Pipeline (triggered by POST /api/upload)

```
Upload MP4  →  OpenCV frame extraction (2s interval)
           →  Fireworks kimi-k2p5 vision model → text descriptions per frame
           →  FFmpeg audio extraction → Fireworks Whisper-v3 → transcripts
           →  Voyage AI or Fireworks GTE-large → embeddings
           →  MongoDB Atlas (frames_col + transcripts_col + video_library_col)
```

### Key Files

| File | Role |
|---|---|
| `llm/config.py` | Single source of truth for all API keys, model IDs, MongoDB clients, and provider selection. Provider is resolved **once at startup** — no per-request switching. |
| `llm/agent.py` | LangGraph `StateGraph` with nodes: `route → retrieve → reason`. ReAct loops back to `retrieve` up to 3× if answer contains "not enough information". |
| `llm/retreival_2.py` | Hybrid search: `hybrid_search_frames()`, `hybrid_search_transcripts()`, `time_windowed_search()`. RRF formula: `score = Σ weight × 1/(60+rank)`. |
| `llm/query_model/router.py` | Fireworks Llama classifies intent into `RouterOutput`. Retries 3× on HTTP 429/5xx; falls back to `FIND_FRAME` on total failure. |
| `llm/query_model/reasoner.py` | `Reasoner.reason()` dispatches to strategy-specific prompt templates. Post-processing extracts timestamps via regex (`\d+\.?\d*s`). |
| `backend/main.py` | FastAPI app. Main endpoints: `POST /api/upload`, `POST /api/chat`, `GET /api/videos`, `GET /api/status/{job_id}`. Ingestion runs in a background thread. |
| `frontend/src/App.tsx` | React 19 + TypeScript SPA. Split-panel: chat (left) + video player (right). Strategy selected via 4 buttons. Timestamps in answers are clickable and seek the video. |
| `evaluation/test_queries.json` | Ground-truth Q&A pairs (10 entries). Must have `reference_answer` field for `eval_matrix.py` to include a query. |

### MongoDB Collections

| Collection | Contents | Vector Dim |
|---|---|---|
| `video_intelligence` (frames) | Frame descriptions + embeddings + `video_id` + `timestamp_seconds` | 1024 (Voyage) or 768 (GTE) |
| `video_intelligence_transcripts` | Audio segments + embeddings + `video_id` + `start_time` | same as above |
| `video_library` | Video metadata: `video_id`, `filename`, `category`, frame count | — |

Text indexes on `description` / `text` fields enable the keyword half of hybrid search.

### Prompting Strategies

| Strategy | Behavior |
|---|---|
| `zero_shot` | Direct 1–2 sentence answer with timestamps |
| `cot` | 4-step chain-of-thought: identify frames → chronological order → discrepancies → answer |
| `few_shot` | Police-report format with 2 in-context exemplars |
| `react` | Same LLM as `cot` but agent loops retrieval up to 3× before answering |

### Evaluation Scripts

`evaluation/eval_matrix.py` is the primary comparison tool: it runs all 4 strategies against `test_queries.json` reference answers and outputs a 4×4 PNG table (BLEU / ROUGE-L / BERTScore / SemScore). Results land in `evaluation/results/`.

BLEU/ROUGE-L are sensitive to format mismatches (CoT/few_shot are verbose; zero_shot is concise), so **SemScore is the most meaningful metric** for comparing strategies.
