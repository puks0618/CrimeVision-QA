from __future__ import annotations
"""
CrimeVision-QA — Query Router

Classifies user queries into 5 intents using Fireworks DeepSeek.
If the API call fails after retries, returns a safe default (FIND_FRAME).
No alternative model fallback — one model, retry on transient errors.
"""

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from llm.config import FIREWORKS_API_KEY, FIREWORKS_API_BASE, FIREWORKS_ROUTER_MODEL

_MAX_RETRIES = 2
_RETRY_DELAYS = [1, 2]

_INTENTS = ["FIND_AUDIO", "FIND_FRAME", "FIND_VIDEO_META", "SUMMARIZE_WINDOW", "COUNT"]
_DEFAULT_INTENT = "FIND_FRAME"

_SYSTEM_PROMPT = """You are a query classifier for a surveillance video QA system.
The video data contains VISUAL FRAMES (images) and optionally AUDIO TRANSCRIPTS.
Classify the user query into exactly ONE intent:

- FIND_FRAME: Default for MOST queries. Use for questions about visuals, appearance, actions,
  scenes, events, descriptions, summaries, and anything about what is VISIBLE in the video.
  Examples: "what happened", "describe the scene", "what are people doing", "who is in the video",
  "describe the incident", "what did the person look like", "summarize the video"

- FIND_AUDIO: ONLY for explicit audio/speech questions: "what was said", "what words",
  "what was heard", "what did they say", "any shouting", "audio transcript"

- FIND_VIDEO_META: Questions about file metadata only: "what is the filename", "how long is the video",
  "what is the resolution", "when was this recorded"

- SUMMARIZE_WINDOW: Questions with explicit time ranges: "between 0:30 and 1:00", "from 30s to 60s"

- COUNT: Counting questions: "how many people", "how many vehicles", "how many times"

Rules:
- When in doubt, use FIND_FRAME (it is the correct default for surveillance analysis)
- "Describe", "What happened", "What is", "Show", "Tell me about" → FIND_FRAME
- Only use FIND_AUDIO if the query explicitly asks about spoken words or audio

Also extract:
- search_query: A concise retrieval-optimized version (remove filler words)
- time_range: {{"start": float, "end": float}} seconds if explicit time window, else null
- confidence: float 0.0-1.0

Respond ONLY with valid JSON. No markdown, no code fences, no explanation.
Example: {{"intent": "FIND_FRAME", "search_query": "people fighting assault", "time_range": null, "confidence": 0.95}}"""


@dataclass
class RouterOutput:
    intent: str
    search_query: str
    time_range: Optional[dict] = None
    confidence: float = 0.5


def _parse_router_response(text: str, original_query: str) -> RouterOutput:
    """Parse the LLM JSON response into a RouterOutput.

    Falls back to safe defaults if parsing fails.
    """
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip()

    try:
        data = json.loads(text)
        intent = data.get("intent", _DEFAULT_INTENT)
        if intent not in _INTENTS:
            intent = _DEFAULT_INTENT

        time_range = data.get("time_range")
        if isinstance(time_range, dict):
            # Ensure both keys exist and are numeric
            if "start" not in time_range or "end" not in time_range:
                time_range = None
            else:
                time_range = {
                    "start": float(time_range["start"]),
                    "end": float(time_range["end"]),
                }
        else:
            time_range = None

        return RouterOutput(
            intent=intent,
            search_query=data.get("search_query", original_query),
            time_range=time_range,
            confidence=float(data.get("confidence", 0.5)),
        )

    except (json.JSONDecodeError, KeyError, ValueError):
        # Attempt regex extraction for intent
        for intent in _INTENTS:
            if intent in text.upper():
                return RouterOutput(
                    intent=intent,
                    search_query=original_query,
                    confidence=0.4,
                )

    return RouterOutput(
        intent=_DEFAULT_INTENT,
        search_query=original_query,
        confidence=0.3,
    )


def route_query(user_query: str, video_id: str | None = None) -> RouterOutput:
    """Classify query intent and extract retrieval parameters.

    Returns a safe default RouterOutput if all API attempts fail.
    """
    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": FIREWORKS_ROUTER_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {user_query}"},
        ],
        "max_tokens": 150,
        "temperature": 0.0,  # deterministic for classification
    }

    for attempt in range(1, _MAX_RETRIES + 2):  # +2 so we get 3 total attempts
        try:
            resp = requests.post(
                f"{FIREWORKS_API_BASE}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            result = _parse_router_response(content, user_query)
            return result

        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response else "?"
            print(f"[Router] HTTP {status} attempt {attempt}")
            if attempt <= _MAX_RETRIES and (status == 429 or (isinstance(status, int) and status >= 500)):
                time.sleep(_RETRY_DELAYS[attempt - 1])
            else:
                break

        except requests.exceptions.RequestException as exc:
            print(f"[Router] Network error attempt {attempt}: {exc}")
            if attempt <= _MAX_RETRIES:
                time.sleep(_RETRY_DELAYS[attempt - 1])

    print(f"[Router] All attempts failed — using default intent '{_DEFAULT_INTENT}'")
    return RouterOutput(
        intent=_DEFAULT_INTENT,
        search_query=user_query,
        confidence=0.0,
    )
