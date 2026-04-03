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

_SYSTEM_PROMPT = """You are a query classifier for a video surveillance QA system.
Classify the user query into exactly ONE of these intents:
- FIND_AUDIO: Questions about what was said, spoken, or heard
- FIND_FRAME: Questions about what was seen, visible, or looked like
- FIND_VIDEO_META: Questions about video metadata (duration, resolution, filename)
- SUMMARIZE_WINDOW: Questions about a specific time range (e.g. "between 0:30 and 1:00")
- COUNT: Questions asking how many (people, vehicles, events)

Also extract:
- search_query: A cleaned version of the query optimized for retrieval
- time_range: {{"start": float, "end": float}} in seconds if a time window is mentioned, else null
- confidence: float 0.0-1.0

Respond ONLY with valid JSON. No markdown, no code fences.
Example: {{"intent": "FIND_FRAME", "search_query": "person in blue jersey", "time_range": null, "confidence": 0.95}}"""


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
