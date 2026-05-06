from __future__ import annotations
"""
CrimeVision-QA — Fine-Tuned Model Evaluation

Compares base Llama-3.1-8B-Instruct vs. fine-tuned (QLoRA adapter) on
ROUGE-L metrics against training-data ground truth.

Usage:
    python finetune/eval_finetuned.py \
        --adapter-path finetune/output/checkpoint-final \
        --video-id Assault001 \
        --training-data finetune/data/training_data.json \
        --output evaluation/results/finetune_comparison.json

REQUIRES GPU.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_TEST_QUERIES = [
    "Generate a police-style incident report for this surveillance footage.",
    "Describe the sequence of events with timestamps.",
    "What criminal activity is depicted?",
]

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 512


def _load_model(model_name_or_path: str, adapter_path: str | None = None):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError:
        print("[Eval] transformers/torch not installed")
        sys.exit(1)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
    )

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"[Eval] Loaded adapter from {adapter_path}")

    model.eval()
    return tokenizer, model


def _generate(tokenizer, model, prompt: str) -> str:
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # greedy decoding — deterministic, no temperature needed
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _compute_rouge(prediction: str, reference: str) -> float:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return scores["rougeL"].fmeasure
    except ImportError:
        return -1.0


def _load_ground_truth(training_data_path: str, video_id: str) -> str | None:
    """Return the expected output for *video_id* from the training data file."""
    if not os.path.exists(training_data_path):
        return None
    with open(training_data_path) as f:
        examples = json.load(f)
    for ex in examples:
        if ex.get("video_id") == video_id:
            return ex.get("output")
    return None


def evaluate(adapter_path: str, video_id: str, training_data_path: str) -> dict:
    from llm.config import frames_col

    # Build prompt context from video frames
    frames = list(
        frames_col.find(
            {"video_id": video_id},
            {"_id": 0, "timestamp_seconds": 1, "description": 1},
        ).sort("timestamp_seconds", 1).limit(10)
    )
    context = "\n".join(
        f"t={f['timestamp_seconds']}s: {f['description']}" for f in frames
    )

    # Load ground-truth reference answer from training data
    ground_truth = _load_ground_truth(training_data_path, video_id)
    if ground_truth:
        print(f"[Eval] Ground truth loaded for video '{video_id}' ({len(ground_truth)} chars)")
    else:
        print(f"[Eval] No ground truth found for '{video_id}' in {training_data_path} — ROUGE will be -1")

    print("[Eval] Loading base model...")
    base_tok, base_model = _load_model(BASE_MODEL)

    print("[Eval] Loading fine-tuned model...")
    ft_tok, ft_model = _load_model(BASE_MODEL, adapter_path)

    results = []
    for query in _TEST_QUERIES:
        prompt = (
            f"### Instruction:\n{query}\n\n"
            f"### Input:\nVideo: {video_id}\n{context}\n\n"
            f"### Response:\n"
        )
        print(f"\n  Query: {query[:60]}...")

        base_answer = _generate(base_tok, base_model, prompt)
        ft_answer = _generate(ft_tok, ft_model, prompt)

        # Compare both answers against training-data ground truth (not against each other)
        rouge_base = _compute_rouge(base_answer, ground_truth) if ground_truth else -1.0
        rouge_ft = _compute_rouge(ft_answer, ground_truth) if ground_truth else -1.0

        results.append(
            {
                "query": query,
                "base_answer": base_answer,
                "finetuned_answer": ft_answer,
                "rouge_l_base": rouge_base,
                "rouge_l_finetuned": rouge_ft,
                "improvement": round(rouge_ft - rouge_base, 4) if ground_truth else None,
            }
        )
        print(f"    Base      ROUGE-L: {rouge_base:.4f}  | {base_answer[:100]}...")
        print(f"    FineTuned ROUGE-L: {rouge_ft:.4f}  | {ft_answer[:100]}...")

    return {
        "video_id": video_id,
        "adapter_path": adapter_path,
        "ground_truth_available": ground_truth is not None,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--training-data", default="finetune/data/training_data.json",
                        help="Path to training_data.json for ground-truth ROUGE reference")
    parser.add_argument("--output", default="evaluation/results/finetune_comparison.json")
    args = parser.parse_args()

    output = evaluate(args.adapter_path, args.video_id, args.training_data)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[Eval] Results saved to {args.output}")


if __name__ == "__main__":
    main()
