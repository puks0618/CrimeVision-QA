from __future__ import annotations
"""
CrimeVision-QA — QLoRA Fine-Tuning Script

Fine-tunes Llama-3.1-8B-Instruct on law enforcement incident report generation
using QLoRA (4-bit quantization + LoRA adapters).

REQUIRES GPU (Google Colab T4/A100 recommended).

Usage:
    python finetune/train_qlora.py \
        --data finetune/data/training_data.json \
        --output finetune/output \
        --epochs 3

Dependencies (install separately on GPU machine):
    pip install torch transformers peft trl bitsandbytes datasets accelerate
"""

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# Guard: check GPU dependencies before importing
# ---------------------------------------------------------------------------
def _check_gpu_deps():
    missing = []
    for pkg in ["torch", "transformers", "peft", "trl", "bitsandbytes", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[QLoRA] Missing GPU dependencies: {', '.join(missing)}")
        print("Install with: pip install torch transformers peft trl bitsandbytes datasets accelerate")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
LEARNING_RATE = 2e-4
MAX_SEQ_LEN = 512


def format_prompt(example: dict) -> str:
    """Format a training example as an instruction-following prompt."""
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )


def train(data_path: str, output_dir: str, epochs: int, batch_size: int) -> None:
    _check_gpu_deps()

    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    # Load training data
    with open(data_path) as f:
        raw_data = json.load(f)

    if not raw_data:
        print(f"[QLoRA] No training examples in {data_path}")
        sys.exit(1)

    # Format examples
    formatted = [{"text": format_prompt(ex)} for ex in raw_data]
    dataset = Dataset.from_list(formatted)
    print(f"[QLoRA] Loaded {len(dataset)} training examples")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load base model
    print(f"[QLoRA] Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training args
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        max_seq_length=MAX_SEQ_LEN,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
    )

    print("[QLoRA] Starting training...")
    trainer.train()

    # Save final adapter
    final_path = os.path.join(output_dir, "checkpoint-final")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"[QLoRA] Adapter saved to {final_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="finetune/data/training_data.json")
    parser.add_argument("--output", default="finetune/output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    train(args.data, args.output, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
