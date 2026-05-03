from __future__ import annotations
"""
CrimeVision-QA — Central Configuration Module

All provider choices are resolved ONCE at import time (not per-request).
Every other module imports from here instead of reading env vars directly.

Usage:
    from llm.config import (
        FIREWORKS_API_KEY, EMBED_PROVIDER, REASONER_PROVIDER,
        frames_col, transcripts_col, video_library_col, incidents_col,
    )
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# ---------------------------------------------------------------------------
# 1. Load .env from project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_env_path = _PROJECT_ROOT / ".env"
load_dotenv(_env_path)

# ---------------------------------------------------------------------------
# 2. Read environment variables
# ---------------------------------------------------------------------------
FIREWORKS_API_KEY: str = os.getenv("FIREWORKS_API_KEY", "")
MONGODB_URI: str = os.getenv("MONGODB_URI", "")
DB_NAME: str = os.getenv("MONGODB_DB_NAME", "video_intelligence")
VOYAGE_API_KEY: str = os.getenv("VOYAGE_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

VIDEO_FOLDER: str = os.getenv("VIDEO_FOLDER", str(_PROJECT_ROOT / "videos"))
FRAMES_FOLDER: str = os.getenv("FRAMES_FOLDER", str(_PROJECT_ROOT / "frames"))

# ---------------------------------------------------------------------------
# 3. Validate REQUIRED keys
# ---------------------------------------------------------------------------
_errors: list[str] = []

if not FIREWORKS_API_KEY:
    _errors.append(
        "FIREWORKS_API_KEY is not set. "
        "Get one at https://fireworks.ai and add it to .env"
    )

if not MONGODB_URI:
    _errors.append(
        "MONGODB_URI is not set. "
        "Add your MongoDB Atlas connection string to .env"
    )

if _errors:
    print("\n[CrimeVision-QA] CONFIGURATION ERROR:")
    for e in _errors:
        print(f"  - {e}")
    print(f"\n  Looked for .env at: {_env_path}\n")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 4. MongoDB connection (singleton)
# ---------------------------------------------------------------------------
_mongo_client: MongoClient = MongoClient(MONGODB_URI)

try:
    _mongo_client.admin.command("ping")
except ConnectionFailure as exc:
    print(f"\n[CrimeVision-QA] MongoDB connection FAILED: {exc}")
    print(f"  URI: {MONGODB_URI[:30]}...")
    sys.exit(1)

_db = _mongo_client[DB_NAME]


def get_db():
    """Return the MongoDB database handle."""
    return _db


# Collection references
frames_col = _db["video_intelligence"]
transcripts_col = _db["video_intelligence_transcripts"]
video_library_col = _db["video_library"]
incidents_col = _db["previous_frame_incidents"]

# ---------------------------------------------------------------------------
# 5. Resolve EMBEDDING provider (once, not per-request)
# ---------------------------------------------------------------------------
_requested_embed = os.getenv("EMBED_PROVIDER", "fireworks").lower()

if _requested_embed == "voyage" and VOYAGE_API_KEY:
    EMBED_PROVIDER = "voyage"
elif _requested_embed == "voyage" and not VOYAGE_API_KEY:
    print("[CONFIG] VOYAGE_API_KEY not set -> using Fireworks GTE-large for embeddings")
    EMBED_PROVIDER = "fireworks"
else:
    EMBED_PROVIDER = "fireworks"

# ---------------------------------------------------------------------------
# 6. Resolve REASONER provider (once, not per-request)
# ---------------------------------------------------------------------------
_requested_reasoner = os.getenv("REASONER_PROVIDER", "gemini").lower()

if _requested_reasoner == "fireworks":
    REASONER_PROVIDER = "fireworks"
elif GEMINI_API_KEY:
    REASONER_PROVIDER = "gemini"
else:
    print("[CONFIG] GEMINI_API_KEY not set -> using Fireworks Llama-3.3-70B for reasoning")
    REASONER_PROVIDER = "fireworks"

# ---------------------------------------------------------------------------
# 7. Model IDs — verified available on this Fireworks account
#    Vision:   kimi-k2p5      — confirmed VLM with image support (tested)
#    Router:   mixtral-8x22b  — reliable JSON output for classification
#    Reasoner: llama-v3p3-70b — best quality chat model available
#    Embed:    gte-large      — 768-dim! Only used when EMBED_PROVIDER=fireworks
#                               (MongoDB index must be 768-dim in that case)
# ---------------------------------------------------------------------------
FIREWORKS_VISION_MODEL = "accounts/fireworks/models/kimi-k2p5"
FIREWORKS_WHISPER_MODEL = "whisper-v3"
FIREWORKS_WHISPER_ENDPOINT = "https://audio-prod.api.fireworks.ai/v1/audio/transcriptions"
FIREWORKS_EMBED_MODEL = "thenlper/gte-large"
FIREWORKS_EMBED_DIM = 768   # GTE-large outputs 768-dim, NOT 1024!
FIREWORKS_ROUTER_MODEL = "accounts/fireworks/models/llama-v3p3-70b-instruct"
FIREWORKS_REASONER_MODEL = "accounts/fireworks/models/llama-v3p3-70b-instruct"
FIREWORKS_API_BASE = "https://api.fireworks.ai/inference/v1"

VOYAGE_EMBED_MODEL = "voyage-3-large"
VOYAGE_EMBED_DIM = 1024  # voyage-3-large outputs 1024-dim
GEMINI_REASONER_MODEL = "gemini-2.5-flash"

# Active embedding dimension (resolved from EMBED_PROVIDER at startup)
EMBED_DIM: int = VOYAGE_EMBED_DIM if EMBED_PROVIDER == "voyage" else FIREWORKS_EMBED_DIM

# ---------------------------------------------------------------------------
# 8. Startup summary
# ---------------------------------------------------------------------------
print("\n[CrimeVision-QA Config]")
print(f"  MongoDB:    Connected ({MONGODB_URI.split('@')[-1].split('/')[0]})")
print(f"  Embeddings: {EMBED_PROVIDER.capitalize()}"
      f" ({'Voyage ' + VOYAGE_EMBED_MODEL if EMBED_PROVIDER == 'voyage' else 'Fireworks ' + FIREWORKS_EMBED_MODEL})"
      f" ({EMBED_DIM}-dim)")
print(f"  Reasoner:   {REASONER_PROVIDER.capitalize()}"
      f" ({'Gemini ' + GEMINI_REASONER_MODEL if REASONER_PROVIDER == 'gemini' else 'Fireworks Llama-3.3-70B'})")
print(f"  Vision:     Fireworks {FIREWORKS_VISION_MODEL.split('/')[-1]}")
print(f"  Audio:      Fireworks {FIREWORKS_WHISPER_MODEL}")
print(f"  Router:     Fireworks {FIREWORKS_ROUTER_MODEL.split('/')[-1]}")
print()
