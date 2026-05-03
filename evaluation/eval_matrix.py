from __future__ import annotations
"""
CrimeVision-QA — Evaluation Matrix v3

Compares prompting strategies (zero_shot, cot, few_shot, react, guided)
against ground-truth reference answers using 7 metrics + latency tracking.

Metrics:
  Text-similarity : BLEU, ROUGE-L, BERTScore, METEOR, SemScore
  Domain-specific  : TimestampF1 (timestamp citation accuracy)
  RAG-quality      : Faithful (answer grounded in retrieved context)

Strategies:
  zero_shot : Direct answer from retrieved evidence
  cot       : Chain-of-thought reasoning then answer
  few_shot  : Style-anchored to example Q→A pairs
  react     : Multi-round retrieval synthesis
  guided    : EVAL-ONLY — reference-guided generation (upper-bound benchmark)

Output:
    - matrix.png        — number table (presentation-ready)
    - matrix_chart.png  — grouped bar chart for visual comparison
    - matrix_radar.png  — radar chart for strategy profiles
    - matrix.json       — full per-query answers + scores + latencies

Usage:
    python evaluation/eval_matrix.py
    python evaluation/eval_matrix.py --limit 2   # smoke test (2 queries)
"""

import argparse
import json
import math
import re
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tqdm import tqdm  # noqa: E402

METRIC_NAMES = ["BLEU", "ROUGE-L", "BERTScore", "METEOR", "SemScore",
                "TimestampF1", "Faithful"]

_FAILURE_STRINGS = ("unable to process", "please try again")

_BERT_SCORER = None
_SEM_MODEL = None
_ROUGE_SCORER = None

def _is_failure(answer: str) -> bool:
    low = (answer or "").lower().strip()
    return not low or any(s in low for s in _FAILURE_STRINGS)


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
            nltk.data.find(f"tokenizers/{resource}" if resource in ("punkt", "punkt_tab")
                           else f"corpora/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Timestamp extraction (reused from reasoner.py pattern)
# ---------------------------------------------------------------------------

_TS_PATTERN_SEC = re.compile(r"(\d+\.?\d*)\s*s(?:ec(?:ond)?s?)?", re.IGNORECASE)
_TS_PATTERN_COLON = re.compile(r"(\d+):(\d{2})(?:\.\d+)?")


def _extract_timestamps_from_text(text: str) -> list[float]:
    """Extract all timestamp references from answer text."""
    timestamps = set()
    for m in _TS_PATTERN_SEC.finditer(text):
        ts = float(m.group(1))
        if ts < 7200:  # sanity: < 2 hours
            timestamps.add(ts)
    for m in _TS_PATTERN_COLON.finditer(text):
        ts = int(m.group(1)) * 60 + int(m.group(2))
        if ts < 7200:
            timestamps.add(float(ts))
    return sorted(timestamps)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def score_bleu(pred: str, ref: str) -> float:
    from nltk.tokenize import word_tokenize
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    if not pred or not ref:
        return 0.0
    pred_t = word_tokenize(pred.lower())
    ref_t = word_tokenize(ref.lower())
    if not pred_t or not ref_t:
        return 0.0
    return float(
        sentence_bleu(
            [ref_t],
            pred_t,
            smoothing_function=SmoothingFunction().method1,
        )
    )


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


def score_timestamp_f1(pred_text: str, expected_timestamps: list[float],
                       tolerance: float = 3.0) -> float:
    """F1 score for timestamp citation accuracy.

    A predicted timestamp is 'matched' if it is within ±tolerance seconds
    of any expected timestamp.
    """
    if not expected_timestamps:
        return float("nan")  # skip queries without expected timestamps

    pred_ts = _extract_timestamps_from_text(pred_text)
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
    recall = len(matched_exp) / len(expected_timestamps) if expected_timestamps else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_faithfulness(answer: str, sources: list[dict]) -> float:
    """Measures how grounded the answer is in retrieved context.

    Uses SemScore between the answer and concatenated source texts.
    High = answer is grounded in context; Low = hallucinated.
    """
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


def _max_score(fn, pred: str, refs: list[str]) -> float:
    """Score pred against each ref and return the maximum."""
    return max((fn(pred, r) for r in refs if r), default=0.0)


def score_all(pred: str, refs: list[str], expected_timestamps: list[float],
              sources: list[dict]) -> dict[str, float]:
    """Score pred against one or more references, taking max per metric.

    `refs` should contain at least the full reference; a concise version may
    also be provided. TimestampF1 and Faithful use the first (full) reference
    for context but don't benefit from dual-scoring.
    """
    if _is_failure(pred):
        return {m: float("nan") for m in METRIC_NAMES}
    refs = [r for r in refs if r]
    if not refs:
        return {m: float("nan") for m in METRIC_NAMES}
    return {
        "BLEU": _max_score(score_bleu, pred, refs),
        "ROUGE-L": _max_score(score_rouge_l, pred, refs),
        "BERTScore": _max_score(score_bertscore, pred, refs),
        "METEOR": _max_score(score_meteor, pred, refs),
        "SemScore": _max_score(score_semscore, pred, refs),
        "TimestampF1": score_timestamp_f1(pred, expected_timestamps),
        "Faithful": score_faithfulness(pred, sources),
    }


# ---------------------------------------------------------------------------
# Agent runner (with 1 retry on failure) — now returns sources + latency
# ---------------------------------------------------------------------------

def run_agent_for(query: str, video_id: str, strategy: str) -> dict:
    """Returns {"answer": str, "sources": list, "latency_s": float}."""
    from llm.agent import run_agent_sync

    latency = 0.0
    for attempt in range(3):
        try:
            t0 = time.perf_counter()
            result = run_agent_sync(query, video_id, strategy)
            latency = round(time.perf_counter() - t0, 2)

            answer = (result or {}).get("answer", "") or ""
            sources = (result or {}).get("sources", []) or []

            if not _is_failure(answer):
                return {"answer": answer, "sources": sources, "latency_s": latency}
            delay = [5, 15][min(attempt, 1)]
            if attempt < 2:
                time.sleep(delay)
        except Exception as e:
            latency = round(time.perf_counter() - t0, 2)
            print(f"  [warn] agent error strategy={strategy} attempt={attempt+1}: {e}",
                  file=sys.stderr)
            if attempt < 2:
                time.sleep(5)
    return {"answer": "", "sources": [], "latency_s": latency}


# ---------------------------------------------------------------------------
# Aggregation + output
# ---------------------------------------------------------------------------

def aggregate(per_query: list[dict], strategies: list[str]) -> dict[str, dict[str, float]]:
    matrix: dict[str, dict[str, float]] = {}
    for strat in strategies:
        per_metric: dict[str, list[float]] = {m: [] for m in METRIC_NAMES}
        latencies: list[float] = []
        for entry in per_query:
            scores = entry["strategy_scores"].get(strat)
            if not scores:
                continue
            for m in METRIC_NAMES:
                v = scores.get(m)
                if v is not None and not math.isnan(v):
                    per_metric[m].append(v)
            lat = entry.get("strategy_latencies", {}).get(strat)
            if lat is not None:
                latencies.append(lat)
        matrix[strat] = {
            m: (sum(vs) / len(vs) if vs else float("nan"))
            for m, vs in per_metric.items()
        }
        matrix[strat]["Avg_Latency_s"] = (
            round(sum(latencies) / len(latencies), 2) if latencies else float("nan")
        )
    return matrix


def print_console(matrix: dict[str, dict[str, float]], strategies: list[str]) -> None:
    all_cols = METRIC_NAMES + ["Avg_Latency_s"]
    col_w = 14
    header = f"{'Strategy':<12}" + "".join(f"{m:>{col_w}}" for m in all_cols)
    print()
    print("=" * len(header))
    print("  EVALUATION MATRIX — Prompting Strategies vs. Ground Truth")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for strat in strategies:
        row = matrix[strat]
        cells = ""
        for m in all_cols:
            v = row.get(m, float("nan"))
            if math.isnan(v):
                cells += f"{'—':>{col_w}}"
            elif m == "Avg_Latency_s":
                cells += f"{v:>{col_w}.2f}"
            else:
                cells += f"{v:>{col_w}.4f}"
        print(f"{strat:<12}{cells}")
    print("=" * len(header))
    print()
    print("Best score per metric (higher is better, except latency):")
    for m in METRIC_NAMES:
        vals = [(s, matrix[s][m]) for s in strategies if not math.isnan(matrix[s].get(m, float("nan")))]
        if vals:
            best = max(vals, key=lambda x: x[1])
            print(f"  {m:<14} → {best[0]} ({best[1]:.4f})")
    # Latency — lower is better
    lat_vals = [(s, matrix[s].get("Avg_Latency_s", float("nan")))
                for s in strategies
                if not math.isnan(matrix[s].get("Avg_Latency_s", float("nan")))]
    if lat_vals:
        best_lat = min(lat_vals, key=lambda x: x[1])
        print(f"  {'Latency (s)':<14} → {best_lat[0]} ({best_lat[1]:.2f}s) ↓ lower is better")
    print()


def save_png(matrix: dict[str, dict[str, float]], strategies: list[str], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_cols = METRIC_NAMES + ["Avg_Latency_s"]
    display_names = METRIC_NAMES + ["Latency(s)"]

    cell_text = []
    for strat in strategies:
        row = matrix[strat]
        cells = []
        for m in all_cols:
            v = row.get(m, float("nan"))
            if math.isnan(v):
                cells.append("—")
            elif m == "Avg_Latency_s":
                cells.append(f"{v:.2f}")
            else:
                cells.append(f"{v:.4f}")
        cell_text.append(cells)

    best_row_per_col: list[int | None] = []
    for i, m in enumerate(all_cols):
        vals = [matrix[s].get(m, float("nan")) for s in strategies]
        finite = [(ri, v) for ri, v in enumerate(vals) if not math.isnan(v)]
        if not finite:
            best_row_per_col.append(None)
        elif m == "Avg_Latency_s":
            best_row_per_col.append(min(finite, key=lambda x: x[1])[0])
        else:
            best_row_per_col.append(max(finite, key=lambda x: x[1])[0])

    fig, ax = plt.subplots(figsize=(14, 0.8 + 0.55 * len(strategies)))
    ax.axis("off")
    ax.set_title(
        "Evaluation Matrix — Prompting Strategies vs. Ground Truth",
        fontsize=13, weight="bold", pad=14,
    )

    table = ax.table(
        cellText=cell_text,
        rowLabels=strategies,
        colLabels=display_names,
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    for ci in range(len(display_names)):
        cell = table[(0, ci)]
        cell.set_facecolor("#1f3a5f")
        cell.set_text_props(color="white", weight="bold")
    for ri in range(len(strategies)):
        cell = table[(ri + 1, -1)]
        cell.set_facecolor("#e8eef7")
        cell.set_text_props(weight="bold")

    for ci, best_ri in enumerate(best_row_per_col):
        if best_ri is None:
            continue
        cell = table[(best_ri + 1, ci)]
        cell.set_facecolor("#d4edda")
        cell.set_text_props(weight="bold")

    fig.text(
        0.5, 0.02,
        "Best score highlighted in green. Higher is better (except Latency). "
        "— = excluded (LLM failure or N/A).",
        ha="center", fontsize=9, style="italic", color="#555",
    )

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_bar_chart(matrix: dict[str, dict[str, float]], strategies: list[str],
                   out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    out_path.parent.mkdir(parents=True, exist_ok=True)

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    n_metrics = len(METRIC_NAMES)
    n_strats = len(strategies)
    bar_w = 0.18
    x = np.arange(n_metrics)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5),
                                    gridspec_kw={"width_ratios": [4, 1]})

    # Quality metrics bar chart
    for si, (strat, color) in enumerate(zip(strategies, colors)):
        vals = [matrix[strat].get(m, float("nan")) for m in METRIC_NAMES]
        offsets = x + (si - n_strats / 2 + 0.5) * bar_w
        bars = ax1.bar(offsets, [0 if math.isnan(v) else v for v in vals],
                      width=bar_w, label=strat, color=color, alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if not math.isnan(v):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7, color="#333")

    ax1.set_xticks(x)
    ax1.set_xticklabels(METRIC_NAMES, fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_title("Quality Metrics by Strategy",
                 fontsize=13, weight="bold", pad=12)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax1.set_axisbelow(True)
    ax1.legend(title="Strategy", fontsize=9, title_fontsize=9,
              loc="upper right", framealpha=0.9)
    ax1.spines[["top", "right"]].set_visible(False)

    # Latency bar chart
    lat_vals = [matrix[s].get("Avg_Latency_s", 0) for s in strategies]
    lat_vals = [0 if math.isnan(v) else v for v in lat_vals]
    bars = ax2.bar(strategies, lat_vals, color=colors[:n_strats], alpha=0.85,
                   edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, lat_vals):
        if v > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{v:.1f}s", ha="center", va="bottom", fontsize=9, color="#333")
    ax2.set_ylabel("Seconds", fontsize=11)
    ax2.set_title("Avg Latency ↓", fontsize=13, weight="bold", pad=12)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(axis="x", rotation=30)

    fig.text(0.5, -0.03,
             "Higher is better for quality metrics. Lower is better for latency. "
             "Missing bars = LLM failure.",
             ha="center", fontsize=9, style="italic", color="#555")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_radar_chart(matrix: dict[str, dict[str, float]], strategies: list[str],
                     out_path: Path) -> None:
    """Spider/radar chart showing each strategy's metric profile."""
    import matplotlib.pyplot as plt
    import numpy as np

    out_path.parent.mkdir(parents=True, exist_ok=True)

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    metrics = METRIC_NAMES  # only quality metrics on radar

    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for si, (strat, color) in enumerate(zip(strategies, colors)):
        vals = []
        for m in metrics:
            v = matrix[strat].get(m, float("nan"))
            vals.append(0 if math.isnan(v) else v)
        vals += vals[:1]  # close polygon
        ax.plot(angles, vals, 'o-', linewidth=2, label=strat, color=color, markersize=5)
        ax.fill(angles, vals, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="#666")
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)

    ax.set_title("Strategy Profile — Radar Chart",
                 fontsize=14, weight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1),
              fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_json(per_query: list[dict], matrix: dict[str, dict[str, float]],
              strategies: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategies": strategies,
        "metrics": METRIC_NAMES,
        "aggregate_matrix": matrix,
        "per_query": per_query,
    }
    out_path.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_queries(path: Path, limit: int | None) -> list[dict]:
    items = json.loads(path.read_text())
    out = []
    for item in items:
        if not item.get("reference_answer"):
            print(f"  [skip] {item.get('query_id')} — no reference_answer", file=sys.stderr)
            continue
        out.append(item)
    if limit is not None:
        out = out[:limit]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path,
                        default=ROOT / "evaluation" / "test_queries.json")
    parser.add_argument("--strategies", type=str,
                        default="zero_shot,cot,few_shot,react,guided",
                        help="Comma-separated list of strategies to compare.")
    parser.add_argument("--out-json", type=Path,
                        default=ROOT / "evaluation" / "results" / "matrix.json")
    parser.add_argument("--out-png", type=Path,
                        default=ROOT / "evaluation" / "results" / "matrix.png")
    parser.add_argument("--out-chart", type=Path,
                        default=ROOT / "evaluation" / "results" / "matrix_chart.png")
    parser.add_argument("--out-radar", type=Path,
                        default=ROOT / "evaluation" / "results" / "matrix_radar.png")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N queries (for smoke testing).")
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    queries = load_queries(args.queries, args.limit)

    if not queries:
        print("No queries with reference_answer to evaluate.", file=sys.stderr)
        return 1

    print(f"Loaded {len(queries)} queries × {len(strategies)} strategies = "
          f"{len(queries) * len(strategies)} agent runs")

    _ensure_nltk()
    print("Loading scoring models...")
    _get_bert_scorer()
    _get_sem_model()
    _get_rouge_scorer()

    per_query: list[dict] = []
    total = len(queries) * len(strategies)
    pbar = tqdm(total=total, desc="Evaluating", unit="run")

    for q in queries:
        entry = {
            "query_id": q.get("query_id"),
            "query": q["query"],
            "video_id": q["video_id"],
            "reference_answer": q["reference_answer"],
            "expected_timestamps": q.get("expected_timestamps", []),
            "strategy_answers": {},
            "strategy_scores": {},
            "strategy_latencies": {},
        }
        expected_ts = q.get("expected_timestamps", [])

        # Build dual-reference list: full + concise (if present)
        refs = [q["reference_answer"]]
        if q.get("reference_answer_concise"):
            refs.append(q["reference_answer_concise"])

        for strat in strategies:
            agent_out = run_agent_for(q["query"], q["video_id"], strat)
            answer = agent_out["answer"]
            sources = agent_out["sources"]
            latency = agent_out["latency_s"]

            entry["strategy_answers"][strat] = answer
            entry["strategy_scores"][strat] = score_all(
                answer, refs, expected_ts, sources
            )
            entry["strategy_latencies"][strat] = latency
            pbar.update(1)
        per_query.append(entry)

    pbar.close()

    matrix = aggregate(per_query, strategies)

    print_console(matrix, strategies)
    save_json(per_query, matrix, strategies, args.out_json)
    save_png(matrix, strategies, args.out_png)
    save_bar_chart(matrix, strategies, args.out_chart)
    save_radar_chart(matrix, strategies, args.out_radar)

    print(f"Wrote: {args.out_json}")
    print(f"Wrote: {args.out_png}")
    print(f"Wrote: {args.out_chart}")
    print(f"Wrote: {args.out_radar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
