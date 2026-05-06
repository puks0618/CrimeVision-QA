from __future__ import annotations
"""
CrimeVision-QA — Embedding Model Fine-Tuning

Fine-tunes BAAI/bge-base-en-v1.5 (768-dim, same as GTE-large) on
surveillance/law-enforcement (query, positive, negative) triplets using
TripletLoss + MultipleNegativesRankingLoss.

The resulting model is a drop-in replacement for the Fireworks GTE-large
embedding path (both output 768-dim vectors — MongoDB index unchanged).

After training, set EMBED_PROVIDER=local and LOCAL_EMBED_PATH in .env.

REQUIRES GPU — but can run on T4 (15GB). A100 is faster.
Estimated time on A100: ~10–20 minutes for 300–500 pairs, 5 epochs.

Usage:
    python finetune/train_embeddings.py \
        --data finetune/data/embedding_training_data.json \
        --output finetune/output/embeddings \
        --epochs 5

Dependencies:
    pip install torch sentence-transformers datasets
"""

import argparse
import json
import os
import sys
from pathlib import Path


def _check_deps():
    missing = []
    for pkg in ["torch", "sentence_transformers", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg.replace("_", "-"))
    if missing:
        print(f"[Embed-FT] Missing dependencies: {', '.join(missing)}")
        print("Install: pip install torch sentence-transformers datasets")
        sys.exit(1)


BASE_MODEL = "BAAI/bge-base-en-v1.5"   # 768-dim — matches existing Fireworks GTE-large path
EMBED_DIM = 768


def train(data_path: str, output_dir: str, epochs: int, batch_size: int) -> None:
    _check_deps()

    from sentence_transformers import SentenceTransformer, InputExample, losses
    from sentence_transformers.evaluation import TripletEvaluator
    from torch.utils.data import DataLoader

    with open(data_path) as f:
        raw_data = json.load(f)

    if not raw_data:
        print(f"[Embed-FT] No training pairs in {data_path}. Run generate_embedding_data.py first.")
        sys.exit(1)

    print(f"[Embed-FT] Loaded {len(raw_data)} triplets")
    print(f"[Embed-FT] Loading base model: {BASE_MODEL}")

    model = SentenceTransformer(BASE_MODEL)

    # Build InputExample list for TripletLoss: (anchor=query, positive, negative)
    train_examples = [
        InputExample(texts=[item["query"], item["positive"], item["negative"]])
        for item in raw_data
    ]

    # 90/10 split
    split = max(1, int(len(train_examples) * 0.9))
    train_set = train_examples[:split]
    eval_set = train_examples[split:]

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)

    # TripletLoss: pushes positive closer to anchor than negative by a margin
    loss = losses.TripletLoss(model=model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=0.5)

    # Evaluator: measures how often positive is closer to anchor than negative
    if eval_set:
        anchors   = [ex.texts[0] for ex in eval_set]
        positives = [ex.texts[1] for ex in eval_set]
        negatives = [ex.texts[2] for ex in eval_set]
        evaluator = TripletEvaluator(anchors, positives, negatives, name="surveillance-triplets")
    else:
        evaluator = None

    warmup_steps = max(1, int(len(train_loader) * epochs * 0.05))
    print(f"[Embed-FT] Training: {epochs} epochs  |  batch={batch_size}  |  warmup={warmup_steps}")

    model.fit(
        train_objectives=[(train_loader, loss)],
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_dir,
        evaluation_steps=max(1, len(train_loader) // 2),
        save_best_model=True,
        show_progress_bar=True,
        optimizer_params={"lr": 2e-5},
        scheduler="WarmupLinear",
    )

    # Save final model
    final_path = os.path.join(output_dir, "checkpoint-final")
    model.save(final_path)
    print(f"[Embed-FT] Final model saved → {final_path}")
    print(f"[Embed-FT] Output dimension: {EMBED_DIM} (compatible with existing MongoDB index)")
    print(f"[Embed-FT] Set EMBED_PROVIDER=local and LOCAL_EMBED_PATH={final_path} in .env to use")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="finetune/data/embedding_training_data.json")
    parser.add_argument("--output", default="finetune/output/embeddings")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    train(args.data, args.output, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
