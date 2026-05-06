from __future__ import annotations
"""
CrimeVision-QA — Master Fine-Tuning Script
===========================================

Run this ONE script on a GPU machine (Google Colab A100 recommended).
It handles everything in order:

  Step 0 — GPU + dependency check
  Step 1 — Install all GPU dependencies
  Step 2 — Generate training data for all 4 components (uses MongoDB)
  Step 3 — Train all 4 models (uses GPU)
  Step 4 — Print .env instructions for using the trained models

Usage (Colab or local GPU):
    python finetune/run_all_finetune.py

Optional flags:
    --skip-frame-describer   Skip Qwen2-VL (longest step, ~2 hrs)
    --skip-whisper           Skip Whisper fine-tuning
    --skip-embeddings        Skip embedding fine-tuning
    --skip-reasoner          Skip Llama-3.1-8B reasoner fine-tuning
    --frames-dir ./frames    Path to extracted frame images (default: ./frames)
    --num-videos 10          Number of videos to use for reasoner training data
    --epochs-vlm 3           Epochs for frame describer (default: 3)
    --epochs-whisper 3       Epochs for Whisper (default: 3)
    --epochs-embeddings 5    Epochs for embeddings (default: 5)
    --epochs-reasoner 3      Epochs for reasoner (default: 3)
    --use-2b-vlm             Use Qwen2-VL-2B instead of 7B (for free Colab T4 15GB)

Estimated total time on A100 (40GB):
    Frame Describer  ~1.5–2.5 hrs
    Whisper          ~20–35 min
    Embeddings       ~10–20 min
    Reasoner         ~25–35 min
    ─────────────────────────────
    Total            ~2.5–4 hrs
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ── Make sure we can import from project root ──────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Output directories ─────────────────────────────────────────────────────
OUT_REASONER   = str(ROOT / "finetune/output/reasoner")
OUT_FRAME      = str(ROOT / "finetune/output/frame_describer")
OUT_WHISPER    = str(ROOT / "finetune/output/whisper")
OUT_EMBEDDINGS = str(ROOT / "finetune/output/embeddings")

DATA_REASONER   = str(ROOT / "finetune/data/training_data.json")
DATA_FRAME      = str(ROOT / "finetune/data/frame_training_data.json")
DATA_WHISPER    = str(ROOT / "finetune/data/whisper_training_data.json")
DATA_EMBEDDINGS = str(ROOT / "finetune/data/embedding_training_data.json")


# ── Helpers ────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}\n")


def _run(label: str, script: str, extra_args: list[str] | None = None) -> bool:
    """Run a finetune sub-script and return True on success."""
    cmd = [sys.executable, str(ROOT / script)] + (extra_args or [])
    _banner(label)
    print(f"Command: {' '.join(cmd)}\n")
    start = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - start
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    if result.returncode == 0:
        print(f"\n[✓] {label} finished in {mins}m {secs}s")
        return True
    else:
        print(f"\n[✗] {label} FAILED (exit code {result.returncode}) after {mins}m {secs}s")
        print(f"    Fix the error above, then re-run with the appropriate --skip-* flags.")
        return False


def _check_gpu() -> bool:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[GPU] {name}  |  VRAM: {vram:.1f} GB")
            return True
        else:
            print("[GPU] CUDA not available — training will be very slow on CPU.")
            ans = input("Continue anyway? [y/N]: ").strip().lower()
            return ans == "y"
    except ImportError:
        print("[GPU] torch not installed yet — will be installed in Step 1.")
        return True


def _install_deps(skip_frame: bool) -> None:
    """Install all GPU fine-tuning dependencies."""
    print(f"Installing base requirements from {ROOT / 'requirements.txt'}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "-r", str(ROOT / "requirements.txt")],
        check=True,
    )

    base = [
        "torch", "transformers", "peft", "trl", "bitsandbytes",
        "datasets", "accelerate", "sentence-transformers",
        "librosa", "soundfile", "jiwer", "evaluate", "rouge_score",
    ]
    vlm_extra = ["qwen-vl-utils"]

    pkgs = base if skip_frame else base + vlm_extra
    print(f"Installing: {', '.join(pkgs)}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade"] + pkgs,
        check=True,
    )
    print("[✓] Dependencies installed")


def _print_env_instructions(
    skip_frame: bool, skip_whisper: bool, skip_embeddings: bool, skip_reasoner: bool
) -> None:
    _banner("DONE — Add these lines to your .env file")

    if not skip_reasoner:
        print(f"# Reasoner: use fine-tuned Llama-3.1-8B instead of Gemini/Fireworks")
        print(f"FINETUNED_ADAPTER_PATH={OUT_REASONER}/checkpoint-final")
        print(f"# Then select 'finetuned' strategy in POST /api/chat\n")

    if not skip_frame:
        print(f"# Frame Describer: use fine-tuned Qwen2-VL instead of kimi-k2p5 API")
        print(f"LOCAL_VLM_ADAPTER_PATH={OUT_FRAME}/checkpoint-final\n")

    if not skip_whisper:
        print(f"# Whisper: use fine-tuned local model instead of Fireworks Whisper-v3 API")
        print(f"LOCAL_WHISPER_PATH={OUT_WHISPER}/checkpoint-final\n")

    if not skip_embeddings:
        print(f"# Embeddings: use fine-tuned BGE-base instead of Fireworks GTE-large API")
        print(f"EMBED_PROVIDER=local")
        print(f"LOCAL_EMBED_PATH={OUT_EMBEDDINGS}/checkpoint-final\n")

    print("Checkpoint locations:")
    for label, path in [
        ("Reasoner",        OUT_REASONER),
        ("Frame Describer", OUT_FRAME),
        ("Whisper",         OUT_WHISPER),
        ("Embeddings",      OUT_EMBEDDINGS),
    ]:
        final = Path(path) / "checkpoint-final"
        status = "✓ exists" if final.exists() else "✗ not found"
        print(f"  {label:<20} {final}  [{status}]")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CrimeVision-QA master fine-tuning script")
    parser.add_argument("--skip-frame-describer",  action="store_true")
    parser.add_argument("--skip-whisper",          action="store_true")
    parser.add_argument("--skip-embeddings",       action="store_true")
    parser.add_argument("--skip-reasoner",         action="store_true")
    parser.add_argument("--frames-dir",            default="./frames")
    parser.add_argument("--num-videos",            type=int, default=10)
    parser.add_argument("--epochs-vlm",            type=int, default=3)
    parser.add_argument("--epochs-whisper",        type=int, default=3)
    parser.add_argument("--epochs-embeddings",     type=int, default=5)
    parser.add_argument("--epochs-reasoner",       type=int, default=3)
    parser.add_argument("--use-2b-vlm",            action="store_true",
                        help="Use Qwen2-VL-2B instead of 7B (free Colab T4)")
    args = parser.parse_args()

    wall_start = time.time()

    _banner("CrimeVision-QA — Master Fine-Tuning Pipeline")
    print(f"Working directory : {ROOT}")
    print(f"Frames directory  : {args.frames_dir}")
    print(f"Skip frame        : {args.skip_frame_describer}")
    print(f"Skip whisper      : {args.skip_whisper}")
    print(f"Skip embeddings   : {args.skip_embeddings}")
    print(f"Skip reasoner     : {args.skip_reasoner}")
    print(f"VLM size          : {'2B (T4 mode)' if args.use_2b_vlm else '7B (A100 mode)'}")

    # ── Step 0: GPU check ────────────────────────────────────────────────
    _banner("Step 0 — GPU Check")
    _check_gpu()

    # ── Step 1: Install dependencies ────────────────────────────────────
    _banner("Step 1 — Installing GPU Dependencies")
    _install_deps(skip_frame=args.skip_frame_describer)

    # ── Patch VLM base model if --use-2b-vlm ────────────────────────────
    if args.use_2b_vlm and not args.skip_frame_describer:
        script_path = ROOT / "finetune/train_frame_describer.py"
        content = script_path.read_text()
        content = content.replace(
            'BASE_MODEL = "Qwen/Qwen2-VL-7B-Instruct"',
            'BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"',
        )
        script_path.write_text(content)
        print("[✓] Patched train_frame_describer.py to use Qwen2-VL-2B-Instruct")

    results: dict[str, bool] = {}

    # ── Step 2 + 3: Generate data → Train (per component) ───────────────

    # ── Embeddings (fastest — run first as warm-up) ──────────────────────
    if not args.skip_embeddings:
        ok = _run(
            "Step 2a — Generate Embedding Training Data",
            "finetune/generate_embedding_data.py",
            ["--output", DATA_EMBEDDINGS],
        )
        if ok:
            results["embeddings"] = _run(
                "Step 3a — Train Embedding Model (BGE-base-en-v1.5)",
                "finetune/train_embeddings.py",
                ["--data", DATA_EMBEDDINGS, "--output", OUT_EMBEDDINGS,
                 "--epochs", str(args.epochs_embeddings)],
            )
        else:
            results["embeddings"] = False

    # ── Whisper ───────────────────────────────────────────────────────────
    if not args.skip_whisper:
        ok = _run(
            "Step 2b — Generate Whisper Training Data",
            "finetune/generate_whisper_data.py",
            ["--frames-dir", args.frames_dir, "--output", DATA_WHISPER],
        )
        if ok:
            results["whisper"] = _run(
                "Step 3b — Fine-Tune Whisper-medium",
                "finetune/train_whisper.py",
                ["--data", DATA_WHISPER, "--output", OUT_WHISPER,
                 "--epochs", str(args.epochs_whisper)],
            )
        else:
            results["whisper"] = False

    # ── Reasoner ──────────────────────────────────────────────────────────
    if not args.skip_reasoner:
        ok = _run(
            "Step 2c — Generate Reasoner Training Data",
            "finetune/generate_training_data.py",
            ["--num-videos", str(args.num_videos), "--output", DATA_REASONER],
        )
        if ok:
            results["reasoner"] = _run(
                "Step 3c — Fine-Tune Reasoner (Llama-3.1-8B QLoRA)",
                "finetune/train_qlora.py",
                ["--data", DATA_REASONER, "--output", OUT_REASONER,
                 "--epochs", str(args.epochs_reasoner)],
            )
        else:
            results["reasoner"] = False

    # ── Frame Describer (longest — run last) ──────────────────────────────
    if not args.skip_frame_describer:
        ok = _run(
            "Step 2d — Generate Frame Describer Training Data",
            "finetune/generate_frame_data.py",
            ["--frames-dir", args.frames_dir, "--output", DATA_FRAME],
        )
        if ok:
            results["frame_describer"] = _run(
                "Step 3d — Fine-Tune Frame Describer (Qwen2-VL QLoRA)",
                "finetune/train_frame_describer.py",
                ["--data", DATA_FRAME, "--output", OUT_FRAME,
                 "--epochs", str(args.epochs_vlm), "--batch-size", "1"],
            )
        else:
            results["frame_describer"] = False

    # ── Summary ───────────────────────────────────────────────────────────
    total_mins = int((time.time() - wall_start) // 60)
    _banner(f"Summary — Total time: {total_mins} minutes")

    all_passed = True
    for name, passed in results.items():
        icon = "✓" if passed else "✗"
        print(f"  [{icon}] {name}")
        if not passed:
            all_passed = False

    if args.skip_frame_describer: print("  [–] frame_describer (skipped)")
    if args.skip_whisper:         print("  [–] whisper (skipped)")
    if args.skip_embeddings:      print("  [–] embeddings (skipped)")
    if args.skip_reasoner:        print("  [–] reasoner (skipped)")

    _print_env_instructions(
        skip_frame=args.skip_frame_describer,
        skip_whisper=args.skip_whisper,
        skip_embeddings=args.skip_embeddings,
        skip_reasoner=args.skip_reasoner,
    )

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
