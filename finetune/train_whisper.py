from __future__ import annotations
"""
CrimeVision-QA — Whisper Fine-Tuning Script

Fine-tunes openai/whisper-medium on law enforcement body-cam / surveillance
audio using (audio_clip, transcript) pairs from generate_whisper_data.py.

After training, set LOCAL_WHISPER_PATH in .env to use this model instead of
the Fireworks Whisper-v3 API.

REQUIRES GPU — Google Colab A100 recommended.
Estimated time on A100: ~20–35 minutes for 200–500 clips, 3 epochs.

Usage:
    python finetune/train_whisper.py \
        --data finetune/data/whisper_training_data.json \
        --output finetune/output/whisper \
        --epochs 3

Dependencies:
    pip install torch transformers datasets librosa soundfile jiwer accelerate
"""

import argparse
import json
import os
import sys
from pathlib import Path


def _check_deps():
    missing = []
    for pkg in ["torch", "transformers", "datasets", "librosa", "soundfile", "jiwer"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[Whisper-FT] Missing dependencies: {', '.join(missing)}")
        print("Install: pip install torch transformers datasets librosa soundfile jiwer accelerate")
        sys.exit(1)


BASE_MODEL = "openai/whisper-medium"
LANGUAGE = "en"
TASK = "transcribe"
MAX_AUDIO_SECONDS = 30   # Whisper's natural window; clips longer than this are truncated


def _load_audio(path: str, sr: int = 16000):
    import librosa
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


def train(data_path: str, output_dir: str, epochs: int, batch_size: int) -> None:
    _check_deps()

    import torch
    import numpy as np
    from transformers import (
        WhisperFeatureExtractor,
        WhisperTokenizer,
        WhisperProcessor,
        WhisperForConditionalGeneration,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )
    import evaluate

    with open(data_path) as f:
        raw_data = json.load(f)

    if not raw_data:
        print(f"[Whisper-FT] No training examples in {data_path}. Run generate_whisper_data.py first.")
        sys.exit(1)

    print(f"[Whisper-FT] Loaded {len(raw_data)} audio/transcript pairs")

    feature_extractor = WhisperFeatureExtractor.from_pretrained(BASE_MODEL)
    tokenizer = WhisperTokenizer.from_pretrained(BASE_MODEL, language=LANGUAGE, task=TASK)
    processor = WhisperProcessor.from_pretrained(BASE_MODEL, language=LANGUAGE, task=TASK)

    print(f"[Whisper-FT] Loading base model: {BASE_MODEL}")
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    # Ensure generation uses English transcription
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None

    # Build HuggingFace dataset from raw_data
    def prepare_example(item: dict) -> dict | None:
        try:
            audio = _load_audio(item["audio_path"])
        except Exception as exc:
            print(f"  [Whisper-FT] Failed to load {item['audio_path']}: {exc}")
            return None

        # Trim / pad to MAX_AUDIO_SECONDS
        max_samples = MAX_AUDIO_SECONDS * 16000
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        input_features = feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features[0]

        labels = tokenizer(item["text"]).input_ids
        return {"input_features": input_features, "labels": labels}

    print("[Whisper-FT] Preparing dataset...")
    prepared = [prepare_example(item) for item in raw_data]
    prepared = [ex for ex in prepared if ex is not None]
    print(f"[Whisper-FT] Prepared {len(prepared)} examples (dropped failed audio)")

    # Train/val split: 90/10
    split = max(1, int(len(prepared) * 0.9))
    train_data = prepared[:split]
    eval_data = prepared[split:] or prepared[:1]  # fallback if tiny dataset

    # Data collator — pads labels with -100 (ignored by loss)
    import dataclasses
    from dataclasses import dataclass
    from typing import Any

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features):
            import torch
            input_features = [{"input_features": f["input_features"]} for f in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            # Remove BOS token prepended by tokenizer (Whisper handles it)
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # WER metric
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    from datasets import Dataset
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=max(1, len(train_data) // batch_size // 5),
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    print("[Whisper-FT] Starting training...")
    trainer.train()

    final_path = os.path.join(output_dir, "checkpoint-final")
    trainer.save_model(final_path)
    processor.save_pretrained(final_path)
    print(f"[Whisper-FT] Model saved → {final_path}")
    print(f"[Whisper-FT] Set LOCAL_WHISPER_PATH={final_path} in .env to use locally")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="finetune/data/whisper_training_data.json")
    parser.add_argument("--output", default="finetune/output/whisper")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    train(args.data, args.output, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
