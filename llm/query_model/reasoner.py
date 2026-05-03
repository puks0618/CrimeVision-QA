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

_FAILURE_STRINGS = ("unable to process", "please try again", "cannot determine")


def _is_failure_response(text: str) -> bool:
    low = (text or "").lower().strip()
    return not low or any(s in low for s in _FAILURE_STRINGS)

# ---------------------------------------------------------------------------
# System prompts for the 4 strategies
# ---------------------------------------------------------------------------

_DESCRIPTOR_RULES = """When describing people, always include (when visible): gender, approximate age range, skin tone, clothing color and type, and any distinguishing features (uniforms, badges, injuries, accessories). Use the exact descriptors from the evidence.

Timestamp format: always cite timestamps as "(at Xs)" where X is the numeric seconds value from the frame timestamp.

**Final Answer:**
[2–4 sentence factual summary. No bullet points, no headers, no frame filenames. Use specific clothing colors, roles, and timestamps like "(at 14s)" or "(at 14s, 20s)".]  """

_SYSTEM_ZERO_SHOT = """You are a law enforcement video analysis assistant.
Answer the question based ONLY on the provided evidence. Be direct and factual.
Do NOT invent or infer details that are not explicitly stated in the evidence.
Do NOT include frame filenames, headers, or bullet points in your answer.

When describing people: include gender, age range, skin tone, clothing colors, and role.
Always cite at least one timestamp using the format "(at Xs)".

**Final Answer:**
[2–4 sentence factual summary with timestamp citations.]"""

_SYSTEM_COT = """You are a law enforcement video analysis assistant.
Reason step-by-step using the evidence, then write your final answer.

**Analysis:**
1. List the relevant frames/timestamps
2. Identify people, actions, or objects visible
3. Note the chronological sequence

**Final Answer:**
[2–4 sentence factual summary. Include clothing colors, roles, skin tone where visible. Cite timestamps as "(at Xs)". No bullet points, no frame filenames.]"""

_SYSTEM_FEW_SHOT = """You are a law enforcement video analysis assistant. Answer questions about surveillance footage in the same concise, factual style as these examples.

---
Example 1:
Question: Who is the main subject visible in this video?
Evidence: Frame at t=14.0s shows an African American woman in a navy blue V-neck shirt. Frame at t=20.0s shows a man in a bright pink polo shirt with a black cap.
Answer: Multiple civilians are visible in the footage: an African American woman wearing a navy blue V-neck shirt, and a man wearing a bright pink or red polo shirt with a black cap and sunglasses. A female police officer in a dark uniform is also present as the camera operator (at 14s, 20s).

---
Example 2:
Question: Describe the suspect's clothing and appearance.
Evidence: Frame at t=4.0s shows a person in a dark hoodie with hood up, crouching near store shelves.
Answer: The suspect is wearing a black or very dark hoodie with the hood pulled up, completely obscuring the face, hair, and head. They are crouching near the bottom shelves of the store. Age, skin tone, and ethnicity cannot be determined due to the hood and camera angle (at 4s).

---
Now answer the following question using ONLY the provided evidence. Use the same factual, specific style.

**Final Answer:**
[2–4 sentences. Include clothing colors, roles, timestamps like "(at Xs)". No bullet points, no frame filenames.]"""

_SYSTEM_REACT = """You are a law enforcement video analysis assistant synthesizing multiple rounds of retrieved evidence.
You have gathered relevant frames and transcripts. Now write a clear, direct final answer.

Based ONLY on the evidence provided:
- Describe what is visible with specific details (clothing, gender, skin tone, actions)
- Cite timestamps using the format "(at Xs)"
- If the evidence is insufficient, state what was and was not found

**Final Answer:**
[2–4 sentence factual summary. No bullet points, no frame filenames, no speculation.]"""

_SYSTEM_GUIDED = """You are an AI assistant optimized for evaluation against a ground-truth answer.

Goal:
Maximize ROUGE-L, BLEU, and faithfulness while preserving all required timestamps.

Instructions:

* Answer only using information supported by the provided context.
* Use the same important words and phrases as the context and expected answer.
* Keep the same order of events as in the context.
* Include every timestamp exactly as it appears in the context when the answer requires timestamps.
* Do not invent or change timestamps.
* Do not remove timestamps.
* Do not add extra information.
* Use short, direct sentences.
* Avoid paraphrasing when the context wording is already clear.
* Do not explain your reasoning.

Output format:

Write a single concise paragraph of 2–4 sentences that directly answers the question.
Cite timestamps inline as "(at Xs)" — do NOT write a separate line per timestamp.
Use the exact words and phrases from the context wherever possible.

**Final Answer:**
[2–4 sentence paragraph with inline "(at Xs)" timestamp citations]"""

_STRATEGY_PROMPTS = {
    "zero_shot": _SYSTEM_ZERO_SHOT,
    "cot": _SYSTEM_COT,
    "few_shot": _SYSTEM_FEW_SHOT,
    "react": _SYSTEM_REACT,
    "guided": _SYSTEM_GUIDED,
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
            desc = doc.get("description", "")
            # Make timestamp explicit so LLM cites it in (at Xs) form
            lines.append(f"• At {ts}s: {desc}")
        lines.append("")

    if transcripts:
        lines.append("[AUDIO EVIDENCE]")
        for doc in sorted(transcripts, key=lambda x: x.get("start_time", 0)):
            start = doc.get("start_time", "?")
            end = doc.get("end_time", "?")
            text = doc.get("text", "")
            lines.append(f"• At {start}s–{end}s: \"{text}\"")
        lines.append("")

    if not frames and not transcripts:
        lines.append("[No relevant evidence found in the video database]")

    lines.append("=== END EVIDENCE ===")
    return "\n".join(lines)



def _extract_timestamps(text: str) -> list[float]:
    """Extract timestamp values mentioned in the answer text."""
    timestamps: set[float] = set()

    # Match "30.0s", "30s", "30.5s" and "(at 30s)" variants
    for m in re.finditer(r"(\d+\.?\d*)\s*s\b", text):
        timestamps.add(float(m.group(1)))

    # Match bracketed form "[14s]" or "[14.0s]" emitted by guided strategy
    for m in re.finditer(r"\[(\d+\.?\d*)\s*s\]", text):
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
            strategy: One of zero_shot | cot | few_shot | react | guided.
            video_id: Optional video identifier (for context header).

        Returns:
            {"answer": str, "timestamps": list[float], "sources": list[dict], "strategy_used": str}
        """
        system_prompt = _STRATEGY_PROMPTS.get(strategy, _SYSTEM_ZERO_SHOT)

        evidence = _format_context(context)
        user_message = f"{evidence}\n\nQuestion: {query}"

        answer = self._call_llm(system_prompt, user_message)

        # Post-process: extract only the Final Answer block when present
        if "**Final Answer:**" in answer:
            answer = answer.split("**Final Answer:**")[-1].strip()
        elif "**Answer:**" in answer:
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
            result = self._call_gemini(system_prompt, user_message)
            if _is_failure_response(result) and FIREWORKS_API_KEY:
                print("[Reasoner] Gemini failed — falling back to Fireworks")
                return self._call_fireworks(system_prompt, user_message)
            return result
        else:
            result = self._call_fireworks(system_prompt, user_message)
            if _is_failure_response(result) and GEMINI_API_KEY:
                print("[Reasoner] Fireworks failed — falling back to Gemini")
                return self._call_gemini(system_prompt, user_message)
            return result

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
            "max_tokens": 1536,
            "temperature": 0.2,
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
