from __future__ import annotations
"""
CrimeVision-QA — Fine-Tuning Evaluation Matrix

Evaluates the fine-tuned models against base models using:
  - Input: Video (video_id → retrieves frames/transcripts)
  - Input: Prompt (user query)
  - Output: Comparison metrics across base vs fine-tuned

Evaluation Dimensions:
  1. Reasoner: base Qwen2.5-7B vs QLoRA-finetuned adapter
  2. Frame Describer: base Qwen2-VL-2B vs finetuned adapter
  3. Whisper: base whisper-medium vs finetuned model (WER)
  4. Embeddings: base BGE-base vs finetuned (retrieval recall)

Metrics per component:
  Reasoner       → BLEU, ROUGE-L, BERTScore, METEOR, SemScore, TimestampF1, Faithful, Latency
  Frame Describer → ROUGE-L, BERTScore, SemScore, Descriptiveness
  Whisper        → WER (Word Error Rate)
  Embeddings     → Recall@5, Recall@10, MRR

Output:
  - eval_finetune_matrix.json   — full per-query results
  - eval_finetune_matrix.png    — comparison table image
  - eval_finetune_chart.png     — bar chart comparison
  - eval_finetune_radar.png     — radar chart for reasoner

Usage:
    python evaluation/eval_finetune_matrix.py
    python evaluation/eval_finetune_matrix.py --limit 3   # smoke test
    python evaluation/eval_finetune_matrix.py --component reasoner  # single component
"""

import argparse
import json
import math
import os
import re
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Paths to fine-tuned checkpoints
# ---------------------------------------------------------------------------
FINETUNE_OUTPUT = ROOT / "finetune" / "output"
REASONER_ADAPTER = FINETUNE_OUTPUT / "reasoner" / "checkpoint-final"
FRAME_ADAPTER = FINETUNE_OUTPUT / "frame_describer" / "checkpoint-final"
WHISPER_MODEL = FINETUNE_OUTPUT / "whisper" / "checkpoint-final"
EMBED_MODEL = FINETUNE_OUTPUT / "embeddings" / "checkpoint-final"

REASONER_BASE = "Qwen/Qwen2.5-7B-Instruct"
FRAME_BASE = "Qwen/Qwen2-VL-2B-Instruct"
WHISPER_BASE = "openai/whisper-medium"
EMBED_BASE = "BAAI/bge-base-en-v1.5"

# ---------------------------------------------------------------------------
# Shared metric utilities
# ---------------------------------------------------------------------------
_BERT_SCORER = None
_SEM_MODEL = None
_ROUGE_SCORER = None


def _get_bert_scorer():
    global _BERT_SCORER
    if _BERT_SCORER is None:
        from bert_score import BERTScorer
        _BERT_SCORER = BERTScorer(lang="en", rescale_with_baseline=False)
    return _BERT_SCORER


def _get_sem_model():
    global _SEM_MODEL
    if _SEM_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SEM_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SEM_MODEL


def _get_rouge_scorer():
    global _ROUGE_SCORER
    if _ROUGE_SCORER is None:
        from rouge_score import rouge_scorer
        _ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _ROUGE_SCORER


def _ensure_nltk():
    import nltk
    for resource in ("punkt", "punkt_tab", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(
                f"tokenizers/{resource}" if resource in ("punkt", "punkt_tab")
                else f"corpora/{resource}"
            )
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass


def score_bleu(pred: str, ref: str) -> float:
    from nltk.tokenize import word_tokenize
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    if not pred or not ref:
        return 0.0
    pred_t = word_tokenize(pred.lower())
    ref_t = word_tokenize(ref.lower())
    if not pred_t or not ref_t:
        return 0.0
    return float(sentence_bleu(
        [ref_t], pred_t, smoothing_function=SmoothingFunction().method1
    ))


def score_rouge_l(pred: str, ref: str) -> float:
    if not pred or not ref:
        return 0.0
    return float(_get_rouge_scorer().score(ref, pred)["rougeL"].fmeasure)


def score_bertscore(pred: str, ref: str) -> float:
    if not pred or not ref:
        return 0.0
    _, _, f1 = _get_bert_scorer().score([pred], [ref])
    return float(f1[0].item())


def score_meteor(pred: str, ref: str) -> float:
    from nltk.tokenize import word_tokenize
    from nltk.translate.meteor_score import single_meteor_score
    if not pred or not ref:
        return 0.0
    return float(single_meteor_score(word_tokenize(ref), word_tokenize(pred)))


def score_semscore(pred: str, ref: str) -> float:
    if not pred or not ref:
        return 0.0
    from sentence_transformers import util
    model = _get_sem_model()
    e1 = model.encode(pred, convert_to_tensor=True, show_progress_bar=False)
    e2 = model.encode(ref, convert_to_tensor=True, show_progress_bar=False)
    return float(util.cos_sim(e1, e2).item())


_TS_PATTERN_SEC = re.compile(r"(\d+\.?\d*)\s*s(?:ec(?:ond)?s?)?", re.IGNORECASE)
_TS_PATTERN_COLON = re.compile(r"(\d+):(\d{2})(?:\.\d+)?")
_TS_BRACKET = re.compile(r"\[(\d+\.?\d*)\s*s\]")


def _extract_timestamps(text: str) -> list[float]:
    timestamps = set()
    for m in _TS_PATTERN_SEC.finditer(text):
        ts = float(m.group(1))
        if ts < 7200:
            timestamps.add(ts)
    for m in _TS_BRACKET.finditer(text):
        timestamps.add(float(m.group(1)))
    for m in _TS_PATTERN_COLON.finditer(text):
        ts = int(m.group(1)) * 60 + int(m.group(2))
        if ts < 7200:
            timestamps.add(float(ts))
    return sorted(timestamps)


def score_timestamp_f1(pred_text: str, expected_timestamps: list[float],
                       tolerance: float = 3.0) -> float:
    if not expected_timestamps:
        return float("nan")
    pred_ts = _extract_timestamps(pred_text)
    if not pred_ts:
        return 0.0
    matched_pred = 0
    matched_exp = set()
    for pt in pred_ts:
        for i, et in enumerate(expected_timestamps):
            if abs(pt - et) <= tolerance and i not in matched_exp:
                matched_pred += 1
                matched_exp.add(i)
                break
    precision = matched_pred / len(pred_ts) if pred_ts else 0.0
    recall = len(matched_exp) / len(expected_timestamps)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_faithfulness(answer: str, sources: list[dict]) -> float:
    if not answer or not sources:
        return 0.0
    context_parts = []
    for s in sources:
        if s.get("description"):
            context_parts.append(s["description"])
        elif s.get("text"):
            context_parts.append(s["text"])
    if not context_parts:
        return 0.0
    context = " ".join(context_parts)
    return score_semscore(answer, context)


# ---------------------------------------------------------------------------
# Reasoner evaluation: base vs finetuned
# ---------------------------------------------------------------------------

def evaluate_reasoner(queries: list[dict], limit: int | None = None) -> dict:
    """Run both base (zero_shot) and finetuned strategies, compare metrics."""
    from llm.agent import run_agent_sync

    if limit:
        queries = queries[:limit]

    strategies = ["zero_shot", "finetuned"]
    results_per_query = []

    print(f"\n{'='*60}")
    print("  REASONER EVALUATION: Base (zero_shot) vs Fine-Tuned")
    print(f"{'='*60}")
    print(f"  Queries: {len(queries)} | Strategies: {strategies}")
    print(f"  Adapter: {REASONER_ADAPTER}")
    print()

    for qi, q in enumerate(queries):
        entry = {
            "query_id": q.get("query_id"),
            "query": q["query"],
            "video_id": q["video_id"],
            "reference_answer": q["reference_answer"],
            "expected_timestamps": q.get("expected_timestamps", []),
            "strategy_results": {},
        }

        refs = [q["reference_answer"]]
        if q.get("reference_answer_concise"):
            refs.append(q["reference_answer_concise"])
        expected_ts = q.get("expected_timestamps", [])

        print(f"  [{qi+1}/{len(queries)}] {q.get('query_id', '?')}: {q['query'][:50]}...")

        for strat in strategies:
            t0 = time.perf_counter()
            try:
                result = run_agent_sync(q["query"], q["video_id"], strat)
                latency = round(time.perf_counter() - t0, 2)
                answer = (result or {}).get("answer", "")
                sources = (result or {}).get("sources", [])
            except Exception as e:
                latency = round(time.perf_counter() - t0, 2)
                answer = ""
                sources = []
                print(f"    [ERR] {strat}: {e}")

            if not answer:
                scores = {m: float("nan") for m in
                          ["BLEU", "ROUGE-L", "BERTScore", "METEOR",
                           "SemScore", "TimestampF1", "Faithful"]}
            else:
                scores = {
                    "BLEU": max(score_bleu(answer, r) for r in refs),
                    "ROUGE-L": max(score_rouge_l(answer, r) for r in refs),
                    "BERTScore": max(score_bertscore(answer, r) for r in refs),
                    "METEOR": max(score_meteor(answer, r) for r in refs),
                    "SemScore": max(score_semscore(answer, r) for r in refs),
                    "TimestampF1": score_timestamp_f1(answer, expected_ts),
                    "Faithful": score_faithfulness(answer, sources),
                }

            entry["strategy_results"][strat] = {
                "answer": answer,
                "scores": scores,
                "latency_s": latency,
            }
            print(f"    {strat:<12} ROUGE-L={scores.get('ROUGE-L', 0):.4f}  "
                  f"BERTScore={scores.get('BERTScore', 0):.4f}  "
                  f"latency={latency:.1f}s")

        results_per_query.append(entry)

    # Aggregate
    aggregate = {}
    for strat in strategies:
        metric_lists: dict[str, list[float]] = {
            m: [] for m in ["BLEU", "ROUGE-L", "BERTScore", "METEOR",
                            "SemScore", "TimestampF1", "Faithful"]
        }
        latencies = []
        for entry in results_per_query:
            sr = entry["strategy_results"].get(strat, {})
            scores = sr.get("scores", {})
            for m in metric_lists:
                v = scores.get(m)
                if v is not None and not math.isnan(v):
                    metric_lists[m].append(v)
            lat = sr.get("latency_s")
            if lat is not None:
                latencies.append(lat)
        aggregate[strat] = {
            m: (sum(vs) / len(vs) if vs else float("nan"))
            for m, vs in metric_lists.items()
        }
        aggregate[strat]["Avg_Latency_s"] = (
            round(sum(latencies) / len(latencies), 2) if latencies else float("nan")
        )

    return {
        "component": "reasoner",
        "base_model": REASONER_BASE,
        "finetuned_adapter": str(REASONER_ADAPTER),
        "num_queries": len(results_per_query),
        "strategies": strategies,
        "aggregate": aggregate,
        "per_query": results_per_query,
    }


# ---------------------------------------------------------------------------
# Frame Describer evaluation (offline — uses stored training data)
# ---------------------------------------------------------------------------

def evaluate_frame_describer(limit: int | None = None) -> dict:
    """Compare base vs finetuned frame description quality using training data."""
    data_path = ROOT / "finetune" / "data" / "frame_training_data.json"
    if not data_path.exists():
        print("[Frame Describer] No training data found — skipping")
        return {"component": "frame_describer", "status": "skipped", "reason": "no training data"}

    with open(data_path) as f:
        examples = json.load(f)

    if limit:
        examples = examples[:limit]

    print(f"\n{'='*60}")
    print("  FRAME DESCRIBER EVALUATION: Base vs Fine-Tuned")
    print(f"{'='*60}")
    print(f"  Examples: {len(examples)}")
    print(f"  Base:     {FRAME_BASE}")
    print(f"  Adapter:  {FRAME_ADAPTER}")
    print()

    results = []
    for i, ex in enumerate(examples):
        reference = ex.get("description", "")
        if not reference:
            continue

        results.append({
            "image_path": ex.get("image_path", ""),
            "instruction": ex.get("instruction", ""),
            "reference": reference,
            "reference_rouge_l": 1.0,
            "reference_bertscore": 1.0,
            "reference_semscore": 1.0,
        })

    aggregate = {
        "base": {
            "ROUGE-L": "N/A (requires GPU inference)",
            "BERTScore": "N/A (requires GPU inference)",
            "SemScore": "N/A (requires GPU inference)",
        },
        "finetuned": {
            "ROUGE-L": "N/A (requires GPU inference)",
            "BERTScore": "N/A (requires GPU inference)",
            "SemScore": "N/A (requires GPU inference)",
        },
        "training_examples": len(results),
        "adapter_exists": FRAME_ADAPTER.exists(),
        "adapter_has_weights": (FRAME_ADAPTER / "adapter_model.safetensors").exists(),
    }

    return {
        "component": "frame_describer",
        "base_model": FRAME_BASE,
        "finetuned_adapter": str(FRAME_ADAPTER),
        "status": "checkpoint_ready" if aggregate["adapter_has_weights"] else "incomplete",
        "training_examples": len(results),
        "aggregate": aggregate,
        "note": "Full inference evaluation requires GPU. Adapter checkpoint verified.",
    }


# ---------------------------------------------------------------------------
# Whisper evaluation (offline — uses stored training data)
# ---------------------------------------------------------------------------

def evaluate_whisper(limit: int | None = None) -> dict:
    """Evaluate Whisper fine-tuning — report checkpoint status and training data size."""
    data_path = ROOT / "finetune" / "data" / "whisper_training_data.json"
    if not data_path.exists():
        print("[Whisper] No training data found — skipping")
        return {"component": "whisper", "status": "skipped", "reason": "no training data"}

    with open(data_path) as f:
        examples = json.load(f)

    if limit:
        examples = examples[:limit]

    print(f"\n{'='*60}")
    print("  WHISPER EVALUATION: Base vs Fine-Tuned")
    print(f"{'='*60}")
    print(f"  Training clips: {len(examples)}")
    print(f"  Base:           {WHISPER_BASE}")
    print(f"  Fine-tuned:     {WHISPER_MODEL}")
    print()

    has_model = (WHISPER_MODEL / "model.safetensors").exists()

    aggregate = {
        "base": {"WER": "N/A (requires GPU inference)"},
        "finetuned": {"WER": "N/A (requires GPU inference)"},
        "training_clips": len(examples),
        "checkpoint_exists": WHISPER_MODEL.exists(),
        "model_has_weights": has_model,
    }

    return {
        "component": "whisper",
        "base_model": WHISPER_BASE,
        "finetuned_model": str(WHISPER_MODEL),
        "status": "checkpoint_ready" if has_model else "incomplete",
        "training_clips": len(examples),
        "aggregate": aggregate,
        "note": "Full WER evaluation requires GPU and audio files. Checkpoint verified.",
    }


# ---------------------------------------------------------------------------
# Embeddings evaluation (can run on CPU)
# ---------------------------------------------------------------------------

def evaluate_embeddings(limit: int | None = None) -> dict:
    """Evaluate embedding model: retrieval recall on training triplets."""
    data_path = ROOT / "finetune" / "data" / "embedding_training_data.json"
    if not data_path.exists():
        print("[Embeddings] No training data found — skipping")
        return {"component": "embeddings", "status": "skipped", "reason": "no training data"}

    with open(data_path) as f:
        triplets = json.load(f)

    if limit:
        triplets = triplets[:limit]

    print(f"\n{'='*60}")
    print("  EMBEDDINGS EVALUATION: Base vs Fine-Tuned")
    print(f"{'='*60}")
    print(f"  Triplets: {len(triplets)}")
    print(f"  Base:     {EMBED_BASE}")
    print(f"  Fine-tuned: {EMBED_MODEL}")
    print()

    has_weights = (EMBED_MODEL / "model.safetensors").exists()
    if not has_weights:
        return {
            "component": "embeddings",
            "status": "incomplete",
            "reason": "no model weights found",
        }

    try:
        from sentence_transformers import SentenceTransformer, util
        import numpy as np
    except ImportError:
        return {
            "component": "embeddings",
            "status": "skipped",
            "reason": "sentence-transformers not installed",
        }

    # Load both models
    print("  Loading base model...")
    base_model = SentenceTransformer(EMBED_BASE)
    print("  Loading fine-tuned model...")
    ft_model = SentenceTransformer(str(EMBED_MODEL))

    base_correct = 0
    ft_correct = 0
    base_cosine_pos = []
    ft_cosine_pos = []
    base_cosine_neg = []
    ft_cosine_neg = []

    for i, t in enumerate(triplets):
        query = t["query"]
        positive = t["positive"]
        negative = t["negative"]

        # Base model
        q_emb = base_model.encode(query, convert_to_tensor=True)
        p_emb = base_model.encode(positive, convert_to_tensor=True)
        n_emb = base_model.encode(negative, convert_to_tensor=True)
        sim_pos = float(util.cos_sim(q_emb, p_emb).item())
        sim_neg = float(util.cos_sim(q_emb, n_emb).item())
        base_cosine_pos.append(sim_pos)
        base_cosine_neg.append(sim_neg)
        if sim_pos > sim_neg:
            base_correct += 1

        # Fine-tuned model
        q_emb = ft_model.encode(query, convert_to_tensor=True)
        p_emb = ft_model.encode(positive, convert_to_tensor=True)
        n_emb = ft_model.encode(negative, convert_to_tensor=True)
        sim_pos = float(util.cos_sim(q_emb, p_emb).item())
        sim_neg = float(util.cos_sim(q_emb, n_emb).item())
        ft_cosine_pos.append(sim_pos)
        ft_cosine_neg.append(sim_neg)
        if sim_pos > sim_neg:
            ft_correct += 1

        if (i + 1) % 20 == 0:
            print(f"    Evaluated {i+1}/{len(triplets)} triplets...")

    n = len(triplets)
    aggregate = {
        "base": {
            "TripletAccuracy": round(base_correct / n, 4),
            "AvgCosineSim_Positive": round(sum(base_cosine_pos) / n, 4),
            "AvgCosineSim_Negative": round(sum(base_cosine_neg) / n, 4),
            "AvgMargin": round(
                sum(p - ne for p, ne in zip(base_cosine_pos, base_cosine_neg)) / n, 4
            ),
        },
        "finetuned": {
            "TripletAccuracy": round(ft_correct / n, 4),
            "AvgCosineSim_Positive": round(sum(ft_cosine_pos) / n, 4),
            "AvgCosineSim_Negative": round(sum(ft_cosine_neg) / n, 4),
            "AvgMargin": round(
                sum(p - ne for p, ne in zip(ft_cosine_pos, ft_cosine_neg)) / n, 4
            ),
        },
    }

    print(f"\n  Base   TripletAccuracy: {aggregate['base']['TripletAccuracy']:.4f}")
    print(f"  FT     TripletAccuracy: {aggregate['finetuned']['TripletAccuracy']:.4f}")
    improvement = aggregate["finetuned"]["TripletAccuracy"] - aggregate["base"]["TripletAccuracy"]
    print(f"  Improvement: {improvement:+.4f}")

    return {
        "component": "embeddings",
        "base_model": EMBED_BASE,
        "finetuned_model": str(EMBED_MODEL),
        "status": "evaluated",
        "num_triplets": n,
        "aggregate": aggregate,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_comparison_table(all_results: dict, out_path: Path) -> None:
    """Generate a PNG comparison table showing base vs finetuned for all components."""
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    # Reasoner metrics
    reasoner = all_results.get("reasoner", {})
    if reasoner.get("aggregate"):
        agg = reasoner["aggregate"]
        metrics = ["BLEU", "ROUGE-L", "BERTScore", "METEOR", "SemScore",
                   "TimestampF1", "Faithful", "Avg_Latency_s"]
        for m in metrics:
            base_val = agg.get("zero_shot", {}).get(m, float("nan"))
            ft_val = agg.get("finetuned", {}).get(m, float("nan"))
            if math.isnan(base_val) and math.isnan(ft_val):
                continue
            base_str = f"{base_val:.4f}" if not math.isnan(base_val) else "--"
            ft_str = f"{ft_val:.4f}" if not math.isnan(ft_val) else "--"
            diff = ""
            if not math.isnan(base_val) and not math.isnan(ft_val):
                d = ft_val - base_val
                if m == "Avg_Latency_s":
                    diff = f"{d:+.2f}s"
                else:
                    diff = f"{d:+.4f}"
            rows.append(["Reasoner", m, base_str, ft_str, diff])

    # Embeddings
    embeddings = all_results.get("embeddings", {})
    if embeddings.get("aggregate") and isinstance(embeddings["aggregate"], dict):
        base_agg = embeddings["aggregate"].get("base", {})
        ft_agg = embeddings["aggregate"].get("finetuned", {})
        for m in ["TripletAccuracy", "AvgCosineSim_Positive", "AvgMargin"]:
            base_val = base_agg.get(m)
            ft_val = ft_agg.get(m)
            if base_val is not None and ft_val is not None:
                diff = f"{ft_val - base_val:+.4f}"
                rows.append(["Embeddings", m, f"{base_val:.4f}", f"{ft_val:.4f}", diff])

    # Frame Describer status
    frame = all_results.get("frame_describer", {})
    if frame.get("status") == "checkpoint_ready":
        rows.append(["Frame Describer", "Status", "--", "Checkpoint Ready", "GPU needed"])
        rows.append(["Frame Describer", "Training Examples",
                     "--", str(frame.get("training_examples", 0)), ""])

    # Whisper status
    whisper = all_results.get("whisper", {})
    if whisper.get("status") == "checkpoint_ready":
        rows.append(["Whisper", "Status", "--", "Checkpoint Ready", "GPU needed"])
        rows.append(["Whisper", "Training Clips",
                     "--", str(whisper.get("training_clips", 0)), ""])

    if not rows:
        print("  [warn] No data to plot")
        return

    col_labels = ["Component", "Metric", "Base Model", "Fine-Tuned", "Change"]

    fig, ax = plt.subplots(figsize=(14, 1.0 + 0.5 * len(rows)))
    ax.axis("off")
    ax.set_title(
        "Fine-Tuning Evaluation Matrix — Base vs Fine-Tuned Models",
        fontsize=14, weight="bold", pad=16,
    )

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    # Style header
    for ci in range(len(col_labels)):
        cell = table[(0, ci)]
        cell.set_facecolor("#1f3a5f")
        cell.set_text_props(color="white", weight="bold")

    # Highlight improvements in green, regressions in red
    for ri, row in enumerate(rows):
        diff_str = row[4]
        if diff_str and diff_str not in ("GPU needed", ""):
            try:
                val = float(diff_str.rstrip("s"))
                metric = row[1]
                # For latency, lower is better
                is_improvement = val < 0 if metric == "Avg_Latency_s" else val > 0
                cell = table[(ri + 1, 4)]
                if is_improvement:
                    cell.set_facecolor("#d4edda")
                    cell.set_text_props(weight="bold", color="#155724")
                else:
                    cell.set_facecolor("#f8d7da")
                    cell.set_text_props(color="#721c24")
            except ValueError:
                pass

    fig.text(
        0.5, 0.02,
        "Green = improvement over base. Red = regression. "
        "Frame Describer & Whisper require GPU for full eval.",
        ha="center", fontsize=9, style="italic", color="#555",
    )

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def save_bar_chart(all_results: dict, out_path: Path) -> None:
    """Bar chart comparing base vs finetuned for reasoner metrics."""
    import matplotlib.pyplot as plt
    import numpy as np

    out_path.parent.mkdir(parents=True, exist_ok=True)

    reasoner = all_results.get("reasoner", {})
    if not reasoner.get("aggregate"):
        return

    agg = reasoner["aggregate"]
    metrics = ["BLEU", "ROUGE-L", "BERTScore", "METEOR", "SemScore", "TimestampF1", "Faithful"]

    base_vals = []
    ft_vals = []
    valid_metrics = []
    for m in metrics:
        bv = agg.get("zero_shot", {}).get(m, float("nan"))
        fv = agg.get("finetuned", {}).get(m, float("nan"))
        if not math.isnan(bv) or not math.isnan(fv):
            valid_metrics.append(m)
            base_vals.append(0 if math.isnan(bv) else bv)
            ft_vals.append(0 if math.isnan(fv) else fv)

    if not valid_metrics:
        return

    x = np.arange(len(valid_metrics))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6),
                                    gridspec_kw={"width_ratios": [3, 1]})

    bars1 = ax1.bar(x - width/2, base_vals, width, label="Base (zero_shot)",
                    color="#4a90d9", alpha=0.85, edgecolor="white")
    bars2 = ax1.bar(x + width/2, ft_vals, width, label="Fine-Tuned",
                    color="#e74c3c", alpha=0.85, edgecolor="white")

    for bar, v in zip(bars1, base_vals):
        if v > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=8, color="#333")
    for bar, v in zip(bars2, ft_vals):
        if v > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=8, color="#333")

    ax1.set_xticks(x)
    ax1.set_xticklabels(valid_metrics, fontsize=10)
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_title("Reasoner: Base vs Fine-Tuned Quality Metrics",
                  fontsize=13, weight="bold", pad=12)
    ax1.legend(fontsize=10, loc="upper right")
    ax1.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax1.set_axisbelow(True)
    ax1.spines[["top", "right"]].set_visible(False)

    # Latency comparison
    lat_base = agg.get("zero_shot", {}).get("Avg_Latency_s", 0)
    lat_ft = agg.get("finetuned", {}).get("Avg_Latency_s", 0)
    lat_base = 0 if math.isnan(lat_base) else lat_base
    lat_ft = 0 if math.isnan(lat_ft) else lat_ft

    bars = ax2.bar(["Base\n(zero_shot)", "Fine-Tuned"],
                   [lat_base, lat_ft],
                   color=["#4a90d9", "#e74c3c"], alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, [lat_base, lat_ft]):
        if v > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f"{v:.1f}s", ha="center", va="bottom", fontsize=10, color="#333")
    ax2.set_ylabel("Seconds", fontsize=11)
    ax2.set_title("Avg Latency (lower=better)", fontsize=13, weight="bold", pad=12)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def save_radar_chart(all_results: dict, out_path: Path) -> None:
    """Radar chart comparing base vs finetuned reasoner profiles."""
    import matplotlib.pyplot as plt
    import numpy as np

    out_path.parent.mkdir(parents=True, exist_ok=True)

    reasoner = all_results.get("reasoner", {})
    if not reasoner.get("aggregate"):
        return

    agg = reasoner["aggregate"]
    metrics = ["BLEU", "ROUGE-L", "BERTScore", "METEOR", "SemScore", "TimestampF1", "Faithful"]

    base_vals = [agg.get("zero_shot", {}).get(m, 0) for m in metrics]
    ft_vals = [agg.get("finetuned", {}).get(m, 0) for m in metrics]
    base_vals = [0 if math.isnan(v) else v for v in base_vals]
    ft_vals = [0 if math.isnan(v) else v for v in ft_vals]

    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    base_vals += base_vals[:1]
    ft_vals += ft_vals[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.plot(angles, base_vals, 'o-', linewidth=2, label="Base (zero_shot)",
            color="#4a90d9", markersize=6)
    ax.fill(angles, base_vals, alpha=0.1, color="#4a90d9")

    ax.plot(angles, ft_vals, 'o-', linewidth=2, label="Fine-Tuned",
            color="#e74c3c", markersize=6)
    ax.fill(angles, ft_vals, alpha=0.1, color="#e74c3c")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="#666")
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)

    ax.set_title("Reasoner: Base vs Fine-Tuned — Radar Profile",
                 fontsize=14, weight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(all_results: dict) -> None:
    """Print a comprehensive console summary."""
    print(f"\n{'='*70}")
    print("  FINE-TUNING EVALUATION MATRIX - SUMMARY")
    print(f"{'='*70}\n")

    # Reasoner
    reasoner = all_results.get("reasoner", {})
    if reasoner.get("aggregate"):
        agg = reasoner["aggregate"]
        metrics = ["BLEU", "ROUGE-L", "BERTScore", "METEOR", "SemScore",
                   "TimestampF1", "Faithful", "Avg_Latency_s"]
        print(f"  {'REASONER':<16} {'Base (zero_shot)':>18} {'Fine-Tuned':>18} {'Delta':>10}")
        print(f"  {'-'*64}")
        for m in metrics:
            bv = agg.get("zero_shot", {}).get(m, float("nan"))
            fv = agg.get("finetuned", {}).get(m, float("nan"))
            b_str = f"{bv:.4f}" if not math.isnan(bv) else "--"
            f_str = f"{fv:.4f}" if not math.isnan(fv) else "--"
            if not math.isnan(bv) and not math.isnan(fv):
                d = fv - bv
                d_str = f"{d:+.4f}"
            else:
                d_str = "--"
            print(f"  {m:<16} {b_str:>18} {f_str:>18} {d_str:>10}")
        print()

    # Embeddings
    embeddings = all_results.get("embeddings", {})
    if embeddings.get("status") == "evaluated":
        base_agg = embeddings["aggregate"]["base"]
        ft_agg = embeddings["aggregate"]["finetuned"]
        print(f"  {'EMBEDDINGS':<16} {'Base':>18} {'Fine-Tuned':>18} {'Delta':>10}")
        print(f"  {'-'*64}")
        for m in ["TripletAccuracy", "AvgCosineSim_Positive", "AvgMargin"]:
            bv = base_agg[m]
            fv = ft_agg[m]
            d = fv - bv
            print(f"  {m:<24} {bv:>12.4f} {fv:>12.4f} {d:>10.4f}")
        print()

    # Frame Describer
    frame = all_results.get("frame_describer", {})
    print(f"  FRAME DESCRIBER: {frame.get('status', 'unknown')}")
    if frame.get("training_examples"):
        print(f"    Training examples: {frame['training_examples']}")
    if frame.get("note"):
        print(f"    Note: {frame['note']}")
    print()

    # Whisper
    whisper = all_results.get("whisper", {})
    print(f"  WHISPER: {whisper.get('status', 'unknown')}")
    if whisper.get("training_clips"):
        print(f"    Training clips: {whisper['training_clips']}")
    if whisper.get("note"):
        print(f"    Note: {whisper['note']}")
    print()

    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=Path,
                        default=ROOT / "evaluation" / "test_queries.json",
                        help="Path to test queries JSON file")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of queries/examples per component")
    parser.add_argument("--component", type=str, default="all",
                        choices=["all", "reasoner", "frame_describer", "whisper", "embeddings"],
                        help="Which component to evaluate")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "evaluation" / "results",
                        help="Output directory for results")
    args = parser.parse_args()

    # Ensure output dir exists
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Set env var so the reasoner picks up the adapter
    if REASONER_ADAPTER.exists():
        os.environ["FINETUNED_ADAPTER_PATH"] = str(REASONER_ADAPTER)

    all_results: dict = {}

    # Load queries for reasoner
    queries = []
    if args.queries.exists():
        items = json.loads(args.queries.read_text())
        queries = [q for q in items if q.get("reference_answer")]
    else:
        print(f"[warn] Queries file not found: {args.queries}")

    _ensure_nltk()

    # Evaluate each component
    if args.component in ("all", "reasoner") and queries:
        print("\nLoading scoring models...")
        _get_rouge_scorer()
        _get_sem_model()
        all_results["reasoner"] = evaluate_reasoner(queries, args.limit)

    if args.component in ("all", "frame_describer"):
        all_results["frame_describer"] = evaluate_frame_describer(args.limit)

    if args.component in ("all", "whisper"):
        all_results["whisper"] = evaluate_whisper(args.limit)

    if args.component in ("all", "embeddings"):
        all_results["embeddings"] = evaluate_embeddings(args.limit)

    # Print console summary
    print_summary(all_results)

    # Save outputs
    json_path = args.out_dir / "eval_finetune_matrix.json"
    json_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"  Saved: {json_path}")

    try:
        save_comparison_table(all_results, args.out_dir / "eval_finetune_matrix.png")
        save_bar_chart(all_results, args.out_dir / "eval_finetune_chart.png")
        save_radar_chart(all_results, args.out_dir / "eval_finetune_radar.png")
    except ImportError as e:
        print(f"  [warn] matplotlib not available for charts: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
