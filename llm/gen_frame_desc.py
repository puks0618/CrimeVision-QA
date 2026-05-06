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
    "Analyze this surveillance frame for criminal activity detection with MAXIMUM detail. "
    "MANDATORY - Report with precision:\n\n"
    "PEOPLE (if visible):\n"
    "- Count, approximate age range, skin tone, ethnicity\n"
    "- Exact clothing: colors, types, visible logos/text, accessories\n"
    "- PRECISE ACTIONS: running/walking/standing/fighting/stealing/concealing/pointing/attacking/fleeing\n"
    "- Hand positions: empty, carrying items, using weapons, raised, in pockets\n"
    "- Visible face details: mask/hoodie obscuring face, visible features, jewelry\n"
    "- Any visible weapons, tools, or stolen items\n"
    "- Injuries, blood, or suspicious marks\n"
    "- Direction of movement with compass direction if possible\n\n"
    "VEHICLES (if visible):\n"
    "- Make, model, color, condition\n"
    "- License plate: any visible digits/characters\n"
    "- Windows tinted/normal, occupants visible\n"
    "- Damage, modifications, unique features\n"
    "- Direction/position relative to scene\n\n"
    "OBJECTS/PROPERTY:\n"
    "- Items on ground, in hands, or being transported\n"
    "- Signs of theft: open doors, broken windows, scattered items\n"
    "- Weapons visible: guns, knives, bats, explosives\n"
    "- Packages, boxes, bags - color and contents if identifiable\n\n"
    "SETTING:\n"
    "- Location type: street/parking lot/store/home/warehouse/alley\n"
    "- Entry/exit points visible\n"
    "- Time of day indicators: sunlight/darkness/street lights\n"
    "- Weather: dry/wet/snowing\n"
    "- Surrounding buildings, signs, landmarks\n\n"
    "CRITICAL OBSERVATIONS:\n"
    "- Is there suspicious behavior? Be specific.\n"
    "- Are multiple people coordinating movements?\n"
    "- Anyone acting as lookout or sentinel?\n"
    "- Signs of organized crime vs opportunistic theft?\n"
    "- Any violence, weapons, or imminent danger indicators?\n\n"
    "IMPORTANT: Be EXTREMELY SPECIFIC and FACTUAL. Avoid generic descriptions like "
    "'appears to show' or 'seems to be'. If you cannot clearly see a detail, write "
    "'[NOT VISIBLE]' instead of guessing. Prioritize details that distinguish this "
    "frame from normal activity."
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
        "max_tokens": 1000,
        "temperature": 0.1,
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


def _validate_description_quality(description: str) -> bool:
    """Check if description is sufficiently detailed (not generic/hallucinated).
    
    Returns True if quality is acceptable, False if suspiciously generic.
    """
    # Reject very short descriptions (likely hallucinated)
    if len(description) < 100:
        return False
    
    # Reject descriptions with explicit uncertainty markers
    if description.count("[NOT VISIBLE]") > 5:
        return False
    
    # Check for minimum specificity — must mention at least some concrete details
    specific_keywords = [
        "color", "clothing", "person", "vehicle", "action", 
        "standing", "running", "walking", "holding", "wearing",
        "number", "street", "store", "parking", "building"
    ]
    
    description_lower = description.lower()
    keyword_count = sum(1 for kw in specific_keywords if kw in description_lower)
    
    # Must have at least 4 specific keywords to avoid generic descriptions
    if keyword_count < 4:
        return False
    
    # Reject descriptions that sound generic
    generic_phrases = [
        "appears to show a scene",
        "shows some people",
        "could be",
        "might be",
        "possibly",
        "unclear what",
        "hard to make out",
    ]
    
    if any(phrase in description_lower for phrase in generic_phrases):
        return False
    
    return True


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
