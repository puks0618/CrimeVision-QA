from __future__ import annotations
"""
CrimeVision-QA — Frame Description via Qwen2.5-VL-32B (Fireworks)

Sends each frame image to the Fireworks vision API and returns a text
description suitable for law-enforcement analysis.
"""

import base64
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm.config import FIREWORKS_API_KEY, FIREWORKS_API_BASE, FIREWORKS_VISION_MODEL

_PROMPT = (
    "Describe this surveillance video frame for law enforcement analysis. "
    "Include: people (appearance, clothing, actions, positions), vehicles "
    "(type, color, partial plates if visible), objects, setting, lighting "
    "conditions, and any visible text or signage. Be concise and factual."
)

_MAX_RETRIES = 3
_RETRY_DELAYS = [1, 2, 4]  # seconds — exponential backoff


def describe_frame(frame_path: str) -> str:
    """Return a text description of the frame at *frame_path*.

    On permanent failure returns '[DESCRIPTION UNAVAILABLE]' and logs the
    error — never crashes the pipeline.
    """
    if not os.path.isfile(frame_path):
        print(f"[Vision] Frame not found: {frame_path}")
        return "[INVALID FRAME]"

    # Detect mime type from extension (dataset uses PNG frames)
    ext = Path(frame_path).suffix.lower()
    mime_type = "image/png" if ext == ".png" else "image/jpeg"

    with open(frame_path, "rb") as f:
        b64_data = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": FIREWORKS_VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{b64_data}"
                        },
                    },
                ],
            }
        ],
        "max_tokens": 300,
        "temperature": 0.2,
    }

    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.post(
                f"{FIREWORKS_API_BASE}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response else "?"
            print(f"[Vision] HTTP {status} on attempt {attempt}/{_MAX_RETRIES}: {frame_path}")
            if status == 429 or (isinstance(status, int) and status >= 500):
                # Rate-limited or server error — retry
                wait = _RETRY_DELAYS[attempt - 1]
                print(f"[Vision] Waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                break  # 4xx client error — no point retrying

        except requests.exceptions.RequestException as exc:
            print(f"[Vision] Network error attempt {attempt}/{_MAX_RETRIES}: {exc}")
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAYS[attempt - 1])

    print(f"[Vision] All retries failed for: {frame_path}")
    return "[DESCRIPTION UNAVAILABLE]"


def describe_frames_batch(
    frame_paths: list[str],
    inter_request_delay: float = 0.5,
) -> list[str]:
    """Describe multiple frames sequentially with rate-limiting delay.

    Returns a list of descriptions in the same order as *frame_paths*.
    """
    descriptions: list[str] = []
    for i, path in enumerate(frame_paths):
        desc = describe_frame(path)
        descriptions.append(desc)
        if i < len(frame_paths) - 1:
            time.sleep(inter_request_delay)
    return descriptions
