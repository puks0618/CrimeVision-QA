from __future__ import annotations
"""
CrimeVision-QA — Reasoner Agent

Generates natural-language answers from retrieved context using one of
4 prompting strategies. Provider (Gemini / Fireworks Llama) is resolved
once at startup — no per-request fallback.
"""

import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from llm.config import (
    FIREWORKS_API_KEY,
    FIREWORKS_API_BASE,
    FIREWORKS_REASONER_MODEL,
    GEMINI_API_KEY,
    GEMINI_REASONER_MODEL,
    REASONER_PROVIDER,
)

_MAX_RETRIES = 2
_RETRY_DELAYS = [2, 4]

# ---------------------------------------------------------------------------
# System prompts for the 4 strategies
# ---------------------------------------------------------------------------

_SYSTEM_ZERO_SHOT = """You are a video surveillance analysis assistant for law enforcement.
Answer the question based ONLY on the provided evidence. Be extremely concise and direct (1-2 sentences).
Always cite specific timestamps (e.g. "at 30.0s") and frame references.
If the evidence does not contain enough information, say so explicitly — do NOT hallucinate."""

_SYSTEM_COT = """You are a video surveillance analysis assistant for law enforcement.
Think step-by-step before answering:
1. Identify all relevant frames and timestamps from the evidence
2. Reconstruct the sequence of events chronologically
3. Note any discrepancies between visual and audio evidence
4. Provide your final answer with timestamp citations

Use this format:
**Analysis:**
[step-by-step reasoning]

**Answer:**
[final answer with timestamps]"""

_SYSTEM_FEW_SHOT = """You are a video surveillance analysis assistant. Format answers like official incident reports.

Example 1:
Q: "What happened at timestamp 0:45?"
A: "At approximately 0:45 (45.0s), Subject A (male, dark hoodie) was observed exiting a white sedan (partial plate: 7X...). Subject proceeded eastbound on foot. [Frames: 0022-0025]"

Example 2:
Q: "Describe the sequence of events."
A: "Timeline of events:
- 0:00-0:15 (0.0s-15.0s): Scene static, parking lot, no activity [Frames: 0001-0007]
- 0:16 (16.0s): Subject A enters frame from north [Frame: 0008]
- 0:23 (23.0s): Subject A approaches vehicle [Frame: 0011]"

Now answer the following question using the same style, based ONLY on the provided evidence:"""

_STRATEGY_PROMPTS = {
    "zero_shot": _SYSTEM_ZERO_SHOT,
    "cot": _SYSTEM_COT,
    "few_shot": _SYSTEM_FEW_SHOT,
    "react": _SYSTEM_COT,  # ReAct uses CoT-style synthesis after tool calls
}


def _format_context(context: list[dict]) -> str:
    """Format retrieved documents into a structured evidence block."""
    frames = [d for d in context if "frame_file" in d or "timestamp_seconds" in d]
    transcripts = [d for d in context if "text" in d and "start_time" in d]

    lines = ["=== RETRIEVED EVIDENCE ===", ""]

    if frames:
        lines.append("[VISUAL EVIDENCE]")
        for doc in sorted(frames, key=lambda x: x.get("timestamp_seconds", 0)):
            ts = doc.get("timestamp_seconds", "?")
            fname = doc.get("frame_file", "unknown")
            desc = doc.get("description", "")
            lines.append(f"- Frame {fname} (t={ts}s): {desc}")
        lines.append("")

    if transcripts:
        lines.append("[AUDIO EVIDENCE]")
        for doc in sorted(transcripts, key=lambda x: x.get("start_time", 0)):
            start = doc.get("start_time", "?")
            end = doc.get("end_time", "?")
            text = doc.get("text", "")
            lines.append(f"- Segment {start}s-{end}s: \"{text}\"")
        lines.append("")

    if not frames and not transcripts:
        lines.append("[No relevant evidence found in the video database]")

    lines.append("=== END EVIDENCE ===")
    return "\n".join(lines)


def _extract_timestamps(text: str) -> list[float]:
    """Extract timestamp values mentioned in the answer text."""
    timestamps: set[float] = set()

    # Match "30.0s", "30s", "30.5s"
    for m in re.finditer(r"(\d+\.?\d*)\s*s\b", text):
        timestamps.add(float(m.group(1)))

    # Match "0:30", "1:05"
    for m in re.finditer(r"\b(\d+):(\d{2})\b", text):
        t = int(m.group(1)) * 60 + int(m.group(2))
        timestamps.add(float(t))

    return sorted(timestamps)


# ---------------------------------------------------------------------------
# Reasoner class
# ---------------------------------------------------------------------------

class Reasoner:
    """Synthesises answers from retrieved evidence using a chosen LLM."""

    def __init__(self) -> None:
        # Provider resolved once at startup in config.py
        self.provider: str = REASONER_PROVIDER

    def reason(
        self,
        query: str,
        context: list[dict],
        strategy: str = "zero_shot",
        video_id: str | None = None,
    ) -> dict:
        """Generate an answer for *query* given *context*.

        Args:
            query:    The user's natural-language question.
            context:  List of retrieved frame/transcript documents.
            strategy: One of zero_shot | cot | few_shot | react.
            video_id: Optional video identifier (for context header).

        Returns:
            {"answer": str, "timestamps": list[float], "sources": list[dict], "strategy_used": str}
        """
        system_prompt = _STRATEGY_PROMPTS.get(strategy, _SYSTEM_ZERO_SHOT)
        evidence = _format_context(context)
        user_message = f"{evidence}\n\nQuestion: {query}"

        answer = self._call_llm(system_prompt, user_message)

        # Post-process to provide only the concise final answer if using CoT
        if "**Answer:**" in answer:
            answer = answer.split("**Answer:**")[-1].strip()

        return {
            "answer": answer,
            "timestamps": _extract_timestamps(answer),
            "sources": context,
            "strategy_used": strategy,
        }

    # ------------------------------------------------------------------
    # LLM dispatch — provider chosen at init, not per-call
    # ------------------------------------------------------------------

    def _call_llm(self, system_prompt: str, user_message: str) -> str:
        if self.provider == "gemini":
            return self._call_gemini(system_prompt, user_message)
        return self._call_fireworks(system_prompt, user_message)

    def _call_gemini(self, system_prompt: str, user_message: str) -> str:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name=GEMINI_REASONER_MODEL,
            system_instruction=system_prompt,
        )

        for attempt in range(1, _MAX_RETRIES + 2):
            try:
                response = model.generate_content(user_message)
                return response.text.strip()
            except Exception as exc:
                print(f"[Reasoner/Gemini] Error attempt {attempt}: {exc}")
                if attempt <= _MAX_RETRIES:
                    time.sleep(_RETRY_DELAYS[attempt - 1])

        return "I was unable to process your query. Please try again."

    def _call_fireworks(self, system_prompt: str, user_message: str) -> str:
        headers = {
            "Authorization": f"Bearer {FIREWORKS_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": FIREWORKS_REASONER_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": 1024,
            "temperature": 0.3,
        }

        for attempt in range(1, _MAX_RETRIES + 2):
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
                print(f"[Reasoner/Fireworks] HTTP {status} attempt {attempt}")
                if attempt <= _MAX_RETRIES:
                    time.sleep(_RETRY_DELAYS[attempt - 1])

            except requests.exceptions.RequestException as exc:
                print(f"[Reasoner/Fireworks] Network error attempt {attempt}: {exc}")
                if attempt <= _MAX_RETRIES:
                    time.sleep(_RETRY_DELAYS[attempt - 1])

        return "I was unable to process your query. Please try again."


# Module-level singleton
reasoner = Reasoner()
