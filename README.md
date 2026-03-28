# SafeGuard AI - CrimeVision-QA

### A Multimodal RAG System for Intelligent Law Enforcement Video Analysis

> **Course:** DATA 266 - Generative Models | Spring 2026 | San Jose State University
> **Group 5:** Sarvesh Waghmare (018319262) | Pukhraj Rathkanthiwar (018274997) | Liza Bharatkumar Lad (018292703)

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Project Summary](#2-project-summary)
3. [System Architecture](#3-system-architecture)
4. [Dataset](#4-dataset)
5. [Technical Approach](#5-technical-approach)
   - 5.1 [Video Ingestion Pipeline](#51-video-ingestion-pipeline)
   - 5.2 [Prompt Engineering (4 Strategies)](#52-prompt-engineering-4-strategies)
   - 5.3 [RAG Pipeline](#53-rag-pipeline)
   - 5.4 [Fine-Tuning (QLoRA / PEFT)](#54-fine-tuning-qlora--peft)
6. [Performance Evaluation](#6-performance-evaluation)
7. [Tech Stack](#7-tech-stack)
8. [Project Structure](#8-project-structure)
9. [Setup & Installation](#9-setup--installation)
10. [API Keys & Configuration](#10-api-keys--configuration)
11. [How to Run](#11-how-to-run)
12. [Evaluation & Results](#12-evaluation--results)
13. [Work Division & Timeline](#13-work-division--timeline)
14. [Reproducibility Guide](#14-reproducibility-guide)
15. [References](#15-references)

---

## 1. Problem Statement

Law enforcement agencies accumulate thousands of hours of body camera and surveillance footage daily. Manual review is labor-intensive, prone to human error, and critically inefficient for time-sensitive investigations. Officers must scrub through hours of video to locate specific events, often missing crucial details that could change the outcome of a case.

**Key challenges:**
- Manual review of 128+ hours of footage per investigation is impractical
- No structured way to query video content using natural language
- Critical visual evidence (weapon drops, vehicle descriptions, suspect movements) is lost in volume
- Police reports are written from memory, missing timestamped visual details

---

## 2. Project Summary

**CrimeVision-QA** is an AI-powered Multimodal Retrieval-Augmented Generation (RAG) system that enables law enforcement professionals to query surveillance and body camera footage using natural language.

**Example queries the system handles:**
- *"When did the suspect discard the weapon?"*
- *"Describe what the person in the blue jersey was doing between 0:30 and 1:00"*
- *"How many people were involved in the incident?"*
- *"What was the sequence of events leading to the collision?"*

The system returns **timestamped, evidence-backed answers** with direct links to the relevant video segments, enabling investigators to go from hours of footage to precise answers in seconds.

### How It Works (End-to-End Flow)

```
Video Input
    |
    v
+-------------------+     +----------------------+     +---------------------+
| Frame Extraction  | --> | Vision-Language Model | --> | Text Descriptions   |
| (OpenCV, 2s int.) |     | (Qwen2.5-VL-32B)     |     | per frame           |
+-------------------+     +----------------------+     +---------------------+
    |                                                           |
    v                                                           v
+-------------------+     +----------------------+     +---------------------+
| Audio Extraction  | --> | Whisper-v3            | --> | Timestamped         |
| (FFmpeg)          |     | (Fireworks API)       |     | Transcripts         |
+-------------------+     +----------------------+     +---------------------+
                                                                |
                                                                v
                                                    +---------------------+
                                                    | Voyage AI / GTE     |
                                                    | Embeddings (1024-d) |
                                                    +---------------------+
                                                                |
                                                                v
                                                    +---------------------+
                                                    | MongoDB Atlas       |
                                                    | Vector Search       |
                                                    | (Hybrid: 70% vec    |
                                                    |  + 30% keyword)     |
                                                    +---------------------+
                                                                |
                                                                v
User Query --> Query Router --> Hybrid Retrieval --> Reasoner Agent --> Timestamped Answer
              (5 intents)      (RRF Scoring)       (Gemini/Llama)     + Video Segment
```

---

## 3. System Architecture

```
safeguard-ai/
|
|-- Frontend (React 19 + TypeScript + Vite)
|       |-- Chat interface with typing animation
|       |-- Split panel: Chat (left) + Video Player (right)
|       |-- Video timestamp seeking from AI responses
|
|-- Backend (FastAPI + Uvicorn)
|       |-- REST API: POST /api/chat
|       |-- Request routing and response orchestration
|       |-- CORS middleware for frontend communication
|
|-- AI/ML Core (LangChain + LangGraph)
|       |-- Query Router (5-intent classification via DeepSeek)
|       |-- Hybrid Retrieval (Vector + Keyword via MongoDB)
|       |-- Reasoner Agent (LLM synthesis with citations)
|       |-- 4 Prompting Strategies (Zero-Shot, CoT, Few-Shot, ReAct)
|
|-- Data Pipeline
|       |-- Frame Extraction (OpenCV, configurable interval)
|       |-- Vision Analysis (Qwen2.5-VL-32B via Fireworks)
|       |-- Audio Transcription (Whisper-v3 via Fireworks)
|       |-- Embedding Generation (Voyage AI / thenlper/gte-large)
|
|-- Database (MongoDB Atlas)
|       |-- video_intelligence (frame metadata + embeddings)
|       |-- video_intelligence_transcripts (audio transcripts + embeddings)
|       |-- video_library (video metadata)
|       |-- previous_frame_incidents (incident tracking)
|       |-- Vector indexes: scalar, binary, full_fidelity (1024-dim, cosine)
|
|-- Fine-Tuning Module
        |-- QLoRA fine-tuning of Llama-3.1-8B-Instruct
        |-- PEFT adapters for law enforcement domain adaptation
        |-- Training on generated incident descriptions
```

---

## 4. Dataset

### 4.1 Primary Dataset: UCF-Crime Dataset

| Property | Details |
|----------|---------|
| **Source** | University of Central Florida (UCF), CVPR 2018 |
| **Paper** | Sultani et al., "Real-World Anomaly Detection in Surveillance Videos" |
| **Total Duration** | ~128 hours of video footage |
| **Total Videos** | 1,900 long, untrimmed surveillance videos |
| **Categories** | 13 anomalous + 1 normal = 14 total |
| **Ground Truth** | Video-level labels (anomaly category) |
| **Resolution** | Variable (real-world CCTV) |
| **Download** | [UCF-Crime Dataset](https://www.crcv.ucf.edu/projects/real-world/) |

### 4.2 Anomaly Categories (13 Types)

| Category | Description |
|----------|-------------|
| Abuse | Physical or verbal mistreatment |
| Arrest | Law enforcement taking suspect into custody |
| Arson | Intentionally setting fire to property |
| Assault | Physical attack on a person |
| Burglary | Illegal entry into a building |
| Explosion | Sudden and violent release of energy |
| Fighting | Physical conflict between individuals |
| Road Accident | Collisions involving vehicles |
| Robbery | Taking property by force or threat |
| Shooting | Use of firearms |
| Shoplifting | Stealing goods from retail store |
| Stealing | General theft or larceny |
| Vandalism | Deliberate destruction of property |

### 4.3 Data Processing Pipeline

```
Raw Video (.mp4)
    |
    +--> OpenCV Frame Extraction (every 2 seconds)
    |       +--> Qwen2.5-VL-32B generates text description per frame
    |       +--> Descriptions embedded via Voyage AI / GTE-Large (1024-dim)
    |       +--> Stored in MongoDB: video_intelligence collection
    |
    +--> FFmpeg Audio Extraction (.mp3)
            +--> Whisper-v3 generates timestamped transcript segments
            +--> Segments embedded via Voyage AI / GTE-Large (1024-dim)
            +--> Stored in MongoDB: video_intelligence_transcripts collection
```

### 4.4 Sample Frame Description (Generated by Qwen2.5-VL)

> **Frame:** `frame_0015_t30.0s.jpg` | **Timestamp:** 30.0s
>
> *"A person wearing a blue jersey is seen running across a parking lot. Two vehicles are parked in the background - a white sedan and a dark SUV. The lighting is dim, suggesting evening or early night. A second individual appears at the far right edge of the frame, partially obscured by a pillar."*

### 4.5 Sample Transcript Segment (Generated by Whisper-v3)

> **Segment:** 28.5s - 35.2s
>
> *"[inaudible]...stop right there...don't move...[siren in background]...get on the ground..."*

---

## 5. Technical Approach

### 5.1 Video Ingestion Pipeline

**Frame Extraction** (`llm/video_to_image.py`):
- Uses OpenCV (`cv2.VideoCapture`) to read video files
- Extracts one frame every `interval_seconds` (default: 2 seconds)
- Saves frames as JPEG with timestamp in filename: `frame_0001_t2.0s.jpg`
- Calculates frame intervals dynamically based on video FPS

**Frame Description Generation** (`llm/gen_frame_desc.py`):
- Encodes each frame to base64
- Sends to Qwen2.5-VL-32B via Fireworks API
- Prompt: *"Describe this video frame. Objects, people, actions, setting, and any visible text. Be concise."*
- Returns max 300 tokens per frame description

**Audio Transcription** (`transcripts/audio.py`):
- Extracts audio from video via FFmpeg (`transcripts/video2audio.py`)
- Sends audio to Fireworks Whisper-v3 endpoint
- Returns verbose JSON with per-segment timestamps
- Endpoint: `audio-prod.us-virginia-1.direct.fireworks.ai/v1/audio/transcriptions`

**Embedding Generation** (`llm/get_voyage_embed.py`):
- Embeds frame descriptions and transcript segments into 1024-dimensional vectors
- Primary provider: Voyage AI (`voyage-multimodal-3`)
- Alternative: Fireworks-hosted `thenlper/gte-large` (same 1024-dim, zero extra API key)
- Implements LRU caching to avoid re-embedding identical text

**MongoDB Storage** (`llm/mongo_client_1.py`):
- 5 collections with vector search indexes
- Supports 3 quantization modes: scalar, binary, full_fidelity
- Cosine similarity for all vector indexes
- Text search indexes on description fields for keyword matching

---

### 5.2 Prompt Engineering (4 Strategies)

All four strategies are implemented in the Reasoner module and compared during evaluation.

#### Strategy 1: Zero-Shot Prompting
Direct question-to-answer without any examples or reasoning scaffolding.
```
System: You are a video analysis assistant for law enforcement.
        Answer based ONLY on the provided evidence.
User:   [Retrieved Context] + [User Question]
```

#### Strategy 2: Chain-of-Thought (CoT) Prompting
Forces the model to reason step-by-step before producing a final answer.
```
System: You are a video analysis assistant for law enforcement.
        Think step-by-step:
        1. First, identify all relevant frames and timestamps
        2. Then, reconstruct the sequence of events chronologically
        3. Note any discrepancies between visual and audio evidence
        4. Finally, provide your answer with timestamp citations
User:   [Retrieved Context] + [User Question]
```

#### Strategy 3: Few-Shot Prompting
Provides 2-3 exemplar Q&A pairs demonstrating the expected output format (police-report style).
```
System: You are a video analysis assistant. Format answers like official incident reports.

Example 1:
  Q: "What happened at timestamp 0:45?"
  A: "At approximately 0:45, Subject A (male, dark hoodie) was observed
      exiting the vehicle (white sedan, partial plate: 7X...). Subject
      proceeded eastbound on foot. [Frames: 0022-0025]"

Example 2:
  Q: "Describe the sequence of events."
  A: "Timeline of events:
      - 0:00-0:15: Scene is static, parking lot, no activity
      - 0:16: Subject A enters frame from the north side
      - 0:23: Subject A approaches vehicle..."

User:   [Retrieved Context] + [User Question]
```

#### Strategy 4: ReAct (Reasoning + Acting)
The LangGraph agent decides WHICH data store to search (visual frames vs. audio transcripts) based on the query type, then reasons over the combined results.
```
Thought: The user is asking about what someone SAID -> search transcripts
Action:  search_transcripts("suspect statement")
Observation: [transcript results with timestamps]
Thought: I also need visual context for the same timeframe -> search frames
Action:  search_frames(timestamp_range="0:30-0:45")
Observation: [frame descriptions for that window]
Thought: I can now synthesize both sources
Answer:  "At 0:32, the suspect stated [transcript]. Visual evidence from
          frame_0016 confirms the suspect was [description]..."
```

---

### 5.3 RAG Pipeline

#### Query Router (`llm/query_model/router.py`)
Classifies incoming queries into 5 intent categories:
| Intent | Example Query | Search Strategy |
|--------|--------------|-----------------|
| `FIND_AUDIO` | "What did the officer say?" | Transcript vector search |
| `FIND_FRAME` | "What color was the car?" | Frame description vector search |
| `FIND_VIDEO_META` | "How long is the video?" | Metadata lookup |
| `SUMMARIZE_WINDOW` | "What happened between 0:30-1:00?" | Time-windowed hybrid search |
| `COUNT` | "How many people were there?" | Frame analysis with counting |

Uses Fireworks-hosted DeepSeek model for intent classification and time-range extraction.

#### Hybrid Retrieval (`llm/retreival_2.py`)
Two implementations for compatibility:

**Native Hybrid Search (MongoDB 8.0+):**
- Uses `$rankFusion` aggregation stage
- Dual pipelines: vector search + text search
- Configurable weights (default: 70% vector, 30% text)

**Manual Hybrid Search (Fallback):**
- Executes vector and text searches independently
- Implements Reciprocal Rank Fusion (RRF) manually
- Formula: `RRF_score = sum(weight * (1 / (60 + rank)))`
- Combines and re-ranks results

**Semantic Search** (`llm/inference.py`):
- Pure vector similarity search via `$vectorSearch` aggregation
- Supports scalar quantization and full fidelity modes
- Multi-collection search: frames + transcripts simultaneously

#### Reasoner Agent (`llm/query_model/reasoner.py`)
- Takes retrieved context (frame descriptions + transcript segments)
- Applies the selected prompting strategy
- Generates natural language answer with timestamp citations
- LLM: Gemini 2.5 Flash (primary) / Llama-3.3-70B via Fireworks (alternative)

---

### 5.4 Fine-Tuning (QLoRA / PEFT)

**Objective:** Adapt an open-source LLM to understand law enforcement terminology and generate police-style incident descriptions superior to the generic base model.

**Method:** Parameter-Efficient Fine-Tuning using QLoRA (Quantized Low-Rank Adaptation)

| Parameter | Value |
|-----------|-------|
| Base Model | `meta-llama/Llama-3.1-8B-Instruct` |
| Quantization | 4-bit (NF4 via bitsandbytes) |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| Target Modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Learning Rate | 2e-4 |
| Epochs | 3 |
| Batch Size | 4 (with gradient accumulation = 4) |
| Max Sequence Length | 512 |
| Training Data | ~200 generated incident descriptions |
| Library | Hugging Face `peft` + `trl` (SFTTrainer) |

**Training Data Generation:**
Incident descriptions are generated by the full RAG pipeline (Qwen VLM + Gemini Reasoner) on processed UCF-Crime videos, then formatted as instruction-response pairs:
```json
{
  "instruction": "Generate a police-style incident report for the following surveillance footage description.",
  "input": "[frame descriptions + transcript segments for a video]",
  "output": "[formatted incident report with timestamps, subject descriptions, sequence of events]"
}
```

**Evaluation:**
The fine-tuned model is compared against the base Llama-3.1-8B-Instruct on the same test queries, measuring ROUGE-L, BERTScore, and human preference scores for domain-specific terminology usage.

---

## 6. Performance Evaluation

### 6.1 Evaluation Metrics

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **Recall@K** | % of relevant frames found in top K retrieval results | > 0.75 at K=5 |
| **RAGAS Faithfulness** | Does the answer stay true to retrieved context? | > 0.80 |
| **RAGAS Answer Relevance** | Does the answer address the question? | > 0.85 |
| **RAGAS Context Precision** | Is the retrieved context relevant? | > 0.70 |
| **Latency** | Time from query to response (seconds) | < 10s |
| **ROUGE-L** | Overlap between generated and reference answers | Fine-tuning comparison |
| **BERTScore** | Semantic similarity of generated vs reference text | Fine-tuning comparison |

### 6.2 Baselines

| Baseline | Description |
|----------|-------------|
| **Baseline 1: Keyword Only** | Naive text search on transcripts (no vector search, no frame data) |
| **Baseline 2: Raw VLM** | Feed frames directly to VLM context window without RAG (no retrieval, limited by context length) |

### 6.3 Comparison Strategy

| Experiment | Retrieval | Prompting | LLM |
|------------|-----------|-----------|-----|
| Baseline 1 | Keyword-only (transcripts) | Zero-Shot | Gemini 2.5 Flash |
| Baseline 2 | None (raw frames to VLM) | Zero-Shot | Qwen2.5-VL-32B |
| Experiment A | Hybrid RAG | Zero-Shot | Gemini 2.5 Flash |
| Experiment B | Hybrid RAG | Chain-of-Thought | Gemini 2.5 Flash |
| Experiment C | Hybrid RAG | Few-Shot | Gemini 2.5 Flash |
| Experiment D | Hybrid RAG (ReAct) | ReAct | Gemini 2.5 Flash |
| Experiment E | Hybrid RAG | Zero-Shot | Fine-tuned Llama-3.1-8B |

---

## 7. Tech Stack

### Backend & AI
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Web Framework | FastAPI + Uvicorn | REST API server |
| Orchestration | LangChain + LangGraph | Agent workflows and RAG pipelines |
| Vision Model | Qwen2.5-VL-32B (Fireworks) | Frame description generation |
| Transcription | Whisper-v3 (Fireworks) | Audio-to-text with timestamps |
| Embeddings | Voyage AI / thenlper/gte-large | 1024-dim vector embeddings |
| Reasoning LLM | Gemini 2.5 Flash / Llama-3.3-70B | Answer synthesis |
| Query Router | DeepSeek (Fireworks) | Intent classification |
| Fine-Tuning | HuggingFace peft + trl + bitsandbytes | QLoRA on Llama-3.1-8B |
| Database | MongoDB Atlas | Vector search + document storage |

### Frontend
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Framework | React 19 | UI components |
| Language | TypeScript 5.9 | Type safety |
| Build Tool | Vite | Fast dev server and bundling |
| Styling | CSS3 | Custom styling |

### Data Processing
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Video Processing | OpenCV (cv2) | Frame extraction |
| Audio Extraction | FFmpeg (libmp3lame) | Video-to-audio conversion |
| Video Download | yt-dlp | YouTube video acquisition |
| Image Handling | Pillow (PIL) | Image encoding/manipulation |

---

## 8. Project Structure

```
safeguard-ai/
|
|-- backend/
|   |-- main.py                    # FastAPI server, /api/chat endpoint
|   |-- requirements.txt           # Backend-specific dependencies
|
|-- frontend/
|   |-- src/
|   |   |-- App.tsx                # Main chat + video player UI
|   |   |-- App.css                # Styling
|   |   |-- main.tsx               # React entry point
|   |-- package.json               # Frontend dependencies
|   |-- vite.config.ts             # Vite configuration
|   |-- tsconfig.json              # TypeScript config
|
|-- llm/
|   |-- agent.py                   # LangGraph agent with ReAct routing
|   |-- inference.py               # Semantic search (vector-only retrieval)
|   |-- retreival_2.py             # Hybrid search (vector + keyword + RRF)
|   |-- mongo_client_1.py          # MongoDB connection + collection setup
|   |-- video_to_image.py          # OpenCV frame extraction
|   |-- gen_frame_desc.py          # Qwen2.5-VL frame description generation
|   |-- get_voyage_embed.py        # Embedding generation (Voyage / GTE)
|   |-- process_frames.py          # Orchestrates frame processing pipeline
|   |-- query_model/
|       |-- router.py              # 5-intent query classification
|       |-- reasoner.py            # LLM answer synthesis (4 prompt strategies)
|
|-- transcripts/
|   |-- video2audio.py             # FFmpeg audio extraction
|   |-- audio.py                   # Whisper-v3 transcription via Fireworks
|
|-- finetune/                      # Fine-tuning module
|   |-- train_qlora.py             # QLoRA training script
|   |-- generate_training_data.py  # Create instruction-response pairs
|   |-- eval_finetuned.py          # Compare base vs fine-tuned model
|
|-- evaluation/
|   |-- eval_retrieval.py          # Recall@K evaluation
|   |-- eval_ragas.py              # RAGAS faithfulness/relevance scores
|   |-- eval_prompts.py            # Compare 4 prompting strategies
|   |-- eval_latency.py            # Latency benchmarks
|   |-- results/                   # Saved evaluation results (JSON/CSV)
|
|-- videos/                        # Raw video files (gitignored)
|-- frames/                        # Extracted frames (gitignored)
|
|-- .env.example                   # Template for environment variables
|-- .gitignore                     # Git ignore rules
|-- requirements.txt               # Root Python dependencies
|-- README.md                      # This file
```

---

## 9. Setup & Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- MongoDB Atlas account (free tier works)
- Fireworks AI API key ([Get one here](https://fireworks.ai))
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/Mongo-db-hackathon/safeguard-ai.git
cd safeguard-ai
```

### Step 2: Backend Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Frontend Setup
```bash
cd frontend
npm install
cd ..
```

### Step 4: MongoDB Atlas Setup
1. Create a free cluster at [MongoDB Atlas](https://www.mongodb.com/atlas)
2. Create a database named `video_intelligence`
3. Run the index setup:
```bash
python llm/mongo_client_1.py
```
This creates all 5 collections and vector search indexes automatically.

### Step 5: Configure Environment Variables
```bash
cp .env.example .env
# Edit .env with your actual API keys (see Section 10)
```

---

## 10. API Keys & Configuration

### `.env.example`
```env
# ============================================
# SafeGuard AI - Environment Configuration
# ============================================

# --- REQUIRED: Fireworks AI (handles Vision, Transcription, Routing, Embeddings) ---
FIREWORKS_API_KEY=fw_your_key_here
# Used for: Qwen2.5-VL (frame descriptions), Whisper-v3 (transcription),
#           DeepSeek (query routing), thenlper/gte-large (embeddings alt)
# Get key: https://fireworks.ai

# --- REQUIRED: MongoDB Atlas ---
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
MONGODB_DB_NAME=video_intelligence

# --- OPTIONAL: Voyage AI (for embeddings - can use Fireworks instead) ---
VOYAGE_API_KEY=pa-your_key_here
# Used for: voyage-multimodal-3 embeddings (1024-dim)
# Alternative: Set EMBED_PROVIDER=fireworks to use thenlper/gte-large instead

# --- OPTIONAL: Google Gemini (for Reasoner LLM - can use Fireworks instead) ---
GEMINI_API_KEY=your_key_here
# Used for: Gemini 2.5 Flash as the Reasoner Agent
# Alternative: Use Fireworks-hosted Llama-3.3-70B

# --- OPTIONAL: OpenAI (legacy - not needed if using Fireworks) ---
# OPENAI_API_KEY=sk-your_key_here

# --- Paths ---
VIDEO_FOLDER=./videos
FRAMES_FOLDER=./frames

# --- Server ---
BACKEND_PORT=8000
FRONTEND_PORT=5173

# --- Embedding Provider Toggle ---
# Options: "voyage" (default) or "fireworks"
EMBED_PROVIDER=voyage
```

### Minimal Setup (1 API Key Only)
If you only have a Fireworks API key, set:
```env
FIREWORKS_API_KEY=fw_your_key_here
MONGODB_URI=mongodb+srv://...
EMBED_PROVIDER=fireworks
```
The system will use Fireworks for ALL AI tasks (vision, transcription, embeddings, routing, and reasoning).

---

## 11. How to Run

### Process a New Video
```bash
# Step 1: Extract frames
python llm/video_to_image.py --video videos/sample.mp4 --output frames/sample/ --interval 2

# Step 2: Generate frame descriptions + embeddings and store in MongoDB
python llm/process_frames.py --frames-dir frames/sample/ --video-id sample_001

# Step 3: Extract and transcribe audio
python transcripts/video2audio.py --video videos/sample.mp4 --output transcripts/sample.mp3
python transcripts/audio.py --audio transcripts/sample.mp3 --video-id sample_001
```

### Start the Application
```bash
# Terminal 1: Start backend
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start frontend
cd frontend
npm run dev
```

### Access the Application
- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:8000
- **API Docs (Swagger):** http://localhost:8000/docs

### Query via API (curl)
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What happened at the 30 second mark?", "video_id": "sample_001"}'
```

---

## 12. Evaluation & Results

### Running Evaluations

```bash
# Evaluate retrieval quality
python evaluation/eval_retrieval.py --test-queries evaluation/test_queries.json

# Evaluate all 4 prompting strategies
python evaluation/eval_prompts.py --strategies zero-shot cot few-shot react

# Run RAGAS evaluation
python evaluation/eval_ragas.py --output evaluation/results/ragas_scores.json

# Benchmark latency
python evaluation/eval_latency.py --queries 20 --output evaluation/results/latency.json

# Compare fine-tuned vs base model
python finetune/eval_finetuned.py --base-model meta-llama/Llama-3.1-8B-Instruct \
                                   --adapter-path finetune/output/checkpoint-final
```

### Results Summary

Results are saved to `evaluation/results/` as JSON and CSV files. See the final project report for detailed analysis, comparison tables, and visualizations.

---

## 13. Work Division & Timeline

### Team Responsibilities

| Member | Role | Responsibilities |
|--------|------|-----------------|
| **Sarvesh Waghmare** | Backend & Data | Video processing pipeline (OpenCV/FFmpeg), MongoDB vector index setup, embedding generation, hybrid retrieval implementation |
| **Pukhraj Rathkanthiwar** | AI Logic | RAG implementation, LangChain/LangGraph agent orchestration, 4 prompting strategies, query router |
| **Liza Bharatkumar Lad** | Fine-tuning & Frontend | QLoRA/PEFT implementation, React frontend, evaluation scripts |

### Timeline & Milestones

| Phase | Dates | Milestone | Deliverable |
|-------|-------|-----------|-------------|
| Phase 1 | Feb 15 - Feb 22 | Data Collection & Setup | UCF-Crime dataset processed, MongoDB Atlas configured |
| Phase 2 | Feb 23 - Mar 05 | Basic RAG Pipeline | **Progress 1:** Working retrieval + single prompting strategy |
| Phase 3 | Mar 06 - Mar 20 | Advanced Features | All 4 prompting strategies + QLoRA fine-tuning complete |
| Phase 4 | Mar 21 - Mar 31 | Integration & Evaluation | Frontend integrated, RAGAS evaluation, comparison tables |
| Phase 5 | Apr 01 - Apr 10 | Final Submission | Report, demo video, code cleanup |

---

## 14. Reproducibility Guide

To reproduce all results from scratch:

1. **Clone and install** (Section 9)
2. **Download UCF-Crime dataset** from [UCF CRCV](https://www.crcv.ucf.edu/projects/real-world/)
3. **Process videos:**
   ```bash
   # Process all videos in batch
   for video in videos/*.mp4; do
     name=$(basename "$video" .mp4)
     python llm/video_to_image.py --video "$video" --output "frames/$name/" --interval 2
     python llm/process_frames.py --frames-dir "frames/$name/" --video-id "$name"
     python transcripts/video2audio.py --video "$video" --output "transcripts/$name.mp3"
     python transcripts/audio.py --audio "transcripts/$name.mp3" --video-id "$name"
   done
   ```
4. **Run evaluation suite:**
   ```bash
   python evaluation/eval_prompts.py --strategies zero-shot cot few-shot react
   python evaluation/eval_ragas.py
   python evaluation/eval_retrieval.py
   ```
5. **Run fine-tuning:**
   ```bash
   python finetune/generate_training_data.py
   python finetune/train_qlora.py
   python finetune/eval_finetuned.py
   ```

### Environment Reproducibility
- Python version: 3.10+
- All dependencies pinned in `requirements.txt`
- Random seeds set in evaluation scripts for deterministic results
- MongoDB indexes are auto-created by `mongo_client_1.py`

---

## 15. References

1. W. Sultani, C. Chen, and M. Shah, "Real-World Anomaly Detection in Surveillance Videos," *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, June 2018.

2. A. Agrawal, J. Lu, S. Antol, M. Mitchell, C. L. Zitnick, D. Batra, and D. Parikh, "VQA: Visual Question Answering," *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2015.

3. S. Lobry, D. Marcos, J. Murray, and D. Tuia, "RSVQA: Visual Question Answering for Remote Sensing Data," *IEEE Transactions on Geoscience and Remote Sensing*, vol. 58, no. 12, 2020.

4. A. Mishra, S. Shekhar, A. K. Singh, and A. Chakraborty, "OCR-VQA: Visual Question Answering by Reading Text in Images," *Proceedings of the 2019 International Conference on Document Analysis and Recognition (ICDAR)*, 2019.

5. P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Kuttler, M. Lewis, W. Yih, T. Rocktaschel, S. Riedel, and D. Kiela, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," *Advances in Neural Information Processing Systems (NeurIPS)*, 2020.

6. E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "LoRA: Low-Rank Adaptation of Large Language Models," *International Conference on Learning Representations (ICLR)*, 2022.

7. T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, "QLoRA: Efficient Finetuning of Quantized Language Models," *Advances in Neural Information Processing Systems (NeurIPS)*, 2023.

8. J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. Chi, Q. V. Le, and D. Zhou, "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models," *Advances in Neural Information Processing Systems (NeurIPS)*, 2022.

9. S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao, "ReAct: Synergizing Reasoning and Acting in Language Models," *International Conference on Learning Representations (ICLR)*, 2023.

10. H. Liu, C. Li, Q. Wu, and Y. J. Lee, "Visual Instruction Tuning (LLaVA)," *Advances in Neural Information Processing Systems (NeurIPS)*, 2023.

11. S. Es, J. James, L. Espinosa-Anke, and S. Schockaert, "RAGAS: Automated Evaluation of Retrieval Augmented Generation," *Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (EACL)*, 2024.

12. J. Johnson, M. Douze, and H. Jegou, "Billion-Scale Similarity Search with GPUs," *IEEE Transactions on Big Data*, vol. 7, no. 3, 2021.

13. A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, "Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)," *Proceedings of the International Conference on Machine Learning (ICML)*, 2023.

14. W. Wang, Q. Lv, W. Yu, W. Hong, J. Qi, Y. Wang, J. Ji, Z. Yang, L. Zhao, X. Song, J. Xu, B. Xu, J. Li, Y. Dong, M. Ding, and J. Tang, "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution," *arXiv preprint arXiv:2409.12191*, 2024.

15. T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, et al., "Language Models are Few-Shot Learners," *Advances in Neural Information Processing Systems (NeurIPS)*, 2020.

16. V. Karpukhin, B. Oguz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, and W. Yih, "Dense Passage Retrieval for Open-Domain Question Answering," *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2020.

17. G. Izacard and E. Grave, "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering," *Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics (EACL)*, 2021.

---

## License

This project is developed for academic purposes as part of DATA 266 - Generative Models, Spring 2026 at San Jose State University.

---

*Built with Fireworks AI, MongoDB Atlas, LangChain, and React.*
