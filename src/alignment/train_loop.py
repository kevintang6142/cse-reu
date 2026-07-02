"""
Outer training loop: self-play → SFT → self-play → SFT → ...

Each iteration:
  1. Self-play generates dialogues using the current model
  2. SFT trains on the generated data (+ existing data)
  3. Repeat

Adapters and self-play artifacts are saved with numbered names:
  adapters/iter_00/lora-adapters/   (initial SFT or base)
  adapters/iter_01/lora-adapters/   (after first self-play → SFT)
  ...
  self_play/iter_01/transcripts.jsonl
  self_play/iter_01/critiques.jsonl
  self_play/iter_01/sft_data.jsonl

Usage:
    python src/scripts/train_loop.py --n-iters 3
    python src/scripts/train_loop.py --n-iters 3 --start-from-base
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import os
import random
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.scripts` resolves
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.alignment.sft import run_sft
from src.alignment.self_play import run_self_play


# ── Defaults ──────────────────────────────────────────────────────────────────
BASE_MODEL_NAME = "Qwen/Qwen3.5-4B"
SSD_ROOT = Path(os.environ.get("SSD_ROOT", "/tmp"))

DEFAULT_MODELS_DIR    = SSD_ROOT / "models"
DEFAULT_SELF_PLAY_DIR = Path("data/self_play")
DEFAULT_TRAIN_DATA    = Path("data/processed/combined_train_chunked.jsonl")
DEFAULT_VAL_DATA      = Path("data/processed/combined_val_chunked.jsonl")
DEFAULT_MODEL_PREFIX  = "Qwen3.5-4B-SFT-CACTUS"

N_ITERS             = 2
N_VIGNETTES         = 300   # distinct CACTUS people sampled per iteration
N_REFINEMENT_CYCLES = 2
SFT_EPOCHS          = 3
API_CONCURRENCY     = 500
THERAPIST_BATCH_SIZE = 8


def run_loop(
    n_iters: int = N_ITERS,
    start_from_base: bool = True,
    models_dir: Path = DEFAULT_MODELS_DIR,
    model_prefix: str = DEFAULT_MODEL_PREFIX,
    self_play_dir: Path = DEFAULT_SELF_PLAY_DIR,
    train_data: Path = DEFAULT_TRAIN_DATA,
    val_data: Path = DEFAULT_VAL_DATA,
    base_model_name: str = BASE_MODEL_NAME,
    n_vignettes: int = N_VIGNETTES,
    n_refinement_cycles: int = N_REFINEMENT_CYCLES,
    sft_epochs: int = SFT_EPOCHS,
    api_concurrency: int = API_CONCURRENCY,
    therapist_batch_size: int = THERAPIST_BATCH_SIZE,
):
    """
    Run the full self-play → SFT loop.

    Models are saved directly in models_dir with names like:
        models_dir/Qwen3.5-4B-SFT-iter_01/
        models_dir/Qwen3.5-4B-SFT-iter_02/
        ...
    Self-play artifacts:
        self_play_dir/iter_01/
        ...
    """
    models_dir = Path(models_dir)
    self_play_dir = Path(self_play_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    self_play_dir.mkdir(parents=True, exist_ok=True)

    # ── Determine starting adapter ────────────────────────────────────────────
    if start_from_base:
        current_adapter = None  # signals "use base model"
        print(f"[Loop] Starting from BASE model: {base_model_name}")
    else:
        # Look for the most recent iteration's model
        existing = sorted(models_dir.glob(f"{model_prefix}-iter_*"))
        if existing:
            current_adapter = str(existing[-1])
            print(f"[Loop] Starting from existing adapter: {current_adapter}")
        else:
            print("[Loop] No existing adapter found — starting from base model")
            current_adapter = None

    # ── Main loop ─────────────────────────────────────────────────────────────
    for iteration in range(1, n_iters + 1):
        print(f"\n{'#'*70}")
        print(f"# ITERATION {iteration}/{n_iters}")
        print(f"# Current adapter: {current_adapter or 'BASE MODEL'}")
        print(f"{'#'*70}\n")

        # ── Step 1: Self-play ─────────────────────────────────────────────────
        sp_dir = self_play_dir / f"iter_{iteration:02d}"
        sp_dir.mkdir(parents=True, exist_ok=True)

        sp_sft_data     = sp_dir / "sft_data.jsonl"
        sp_transcripts  = sp_dir / "transcripts.jsonl"
        sp_critiques    = sp_dir / "critiques.jsonl"

        print(f"\n[Loop] Step 1: Self-play → {sp_dir}")
        run_self_play(
            adapter_path=current_adapter,
            output_jsonl=sp_sft_data,
            output_transcripts=sp_transcripts,
            output_critiques=sp_critiques,
            base_model_name=base_model_name,
            n_vignettes=n_vignettes,
            vignette_seed=42 + iteration,
            n_refinement_cycles=n_refinement_cycles,
            api_concurrency=api_concurrency,
            therapist_batch_size=therapist_batch_size,
        )

        # ── Step 2: SFT on self-play data + the full real (combined) datasets ──
        # AMIE-style recipe: the full real corpus anchors EVERY iteration, with the
        # iteration's self-play dialogues layered on top of it.
        iter_output_dir = models_dir / f"{model_prefix}-iter_{iteration:02d}"

        # Split self-play data 90/10 into train/val
        with open(sp_sft_data, encoding="utf-8") as f:
            sp_lines = [l.strip() for l in f if l.strip()]
        random.seed(42 + iteration)
        random.shuffle(sp_lines)
        split_idx = max(1, int(len(sp_lines) * 0.9))
        sp_train_lines = sp_lines[:split_idx]
        sp_val_lines = sp_lines[split_idx:]

        # Self-play-only artifacts (kept for inspection / debugging)
        sp_train_path = sp_dir / "sft_train.jsonl"
        sp_val_path = sp_dir / "sft_val.jsonl"
        sp_train_path.write_text("\n".join(sp_train_lines) + "\n", encoding="utf-8")
        sp_val_path.write_text("\n".join(sp_val_lines) + "\n", encoding="utf-8")

        # Full real (combined) datasets — mixed in on every iteration
        with open(train_data, encoding="utf-8") as f:
            real_train_lines = [l.strip() for l in f if l.strip()]
        with open(val_data, encoding="utf-8") as f:
            real_val_lines = [l.strip() for l in f if l.strip()]

        # Merge real + self-play, shuffle, and write the files SFT actually trains on
        merged_train_lines = sp_train_lines + real_train_lines
        merged_val_lines = sp_val_lines + real_val_lines
        random.shuffle(merged_train_lines)
        random.shuffle(merged_val_lines)

        train_path = sp_dir / "train.jsonl"
        val_path = sp_dir / "val.jsonl"
        train_path.write_text("\n".join(merged_train_lines) + "\n", encoding="utf-8")
        val_path.write_text("\n".join(merged_val_lines) + "\n", encoding="utf-8")

        print(f"\n[Loop] Step 2: SFT training → {iter_output_dir}")
        print(f"  Train: {len(merged_train_lines)} examples "
              f"({len(sp_train_lines)} self-play + {len(real_train_lines)} real)")
        print(f"  Val:   {len(merged_val_lines)} examples "
              f"({len(sp_val_lines)} self-play + {len(real_val_lines)} real)")

        adapter_path = run_sft(
            train_data=train_path,
            val_data=val_path,
            output_dir=iter_output_dir,
            model_name=base_model_name,
            num_epochs=sft_epochs,
        )

        current_adapter = str(adapter_path)
        print(f"\n[Loop] Iteration {iteration} complete. Adapter: {current_adapter}")

    print(f"\n{'='*70}")
    print(f"TRAINING LOOP COMPLETE — {n_iters} iterations")
    print(f"Final adapter: {current_adapter}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-play → SFT training loop")
    parser.add_argument("--n-iters", type=int, default=N_ITERS,
                        help="Number of self-play → SFT iterations")
    parser.add_argument("--start-from-adapter", action="store_true",
                        help="Start from existing adapter instead of base model")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR,
                        help="Directory for saving models (flat, no subfolders)")
    parser.add_argument("--model-prefix", default=DEFAULT_MODEL_PREFIX,
                        help="Model name prefix (e.g. Qwen3.5-4B-SFT)")
    parser.add_argument("--self-play-dir", type=Path, default=DEFAULT_SELF_PLAY_DIR,
                        help="Directory for self-play outputs")
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA,
                        help="Path to base training data JSONL")
    parser.add_argument("--val-data", type=Path, default=DEFAULT_VAL_DATA,
                        help="Path to validation data JSONL")
    parser.add_argument("--base-model", default=BASE_MODEL_NAME)
    parser.add_argument("--n-vignettes", type=int, default=N_VIGNETTES,
                        help="Distinct CACTUS people sampled per iteration")
    parser.add_argument("--n-refinement-cycles", type=int, default=N_REFINEMENT_CYCLES)
    parser.add_argument("--sft-epochs", type=int, default=SFT_EPOCHS)
    parser.add_argument("--api-concurrency", type=int, default=API_CONCURRENCY)
    parser.add_argument("--therapist-batch-size", type=int, default=THERAPIST_BATCH_SIZE,
                        help="Batch size for local HuggingFace therapist model.generate calls")
    args = parser.parse_args()

    run_loop(
        n_iters=args.n_iters,
        start_from_base=not args.start_from_adapter,
        models_dir=args.models_dir,
        model_prefix=args.model_prefix,
        self_play_dir=args.self_play_dir,
        train_data=args.train_data,
        val_data=args.val_data,
        base_model_name=args.base_model,
        n_vignettes=args.n_vignettes,
        n_refinement_cycles=args.n_refinement_cycles,
        sft_epochs=args.sft_epochs,
        api_concurrency=args.api_concurrency,
        therapist_batch_size=args.therapist_batch_size,
    )
