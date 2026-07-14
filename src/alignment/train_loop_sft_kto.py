"""
Outer training loop (SFT+KTO variant): self-play → SFT → KTO → repeat.

Combines src/alignment/train_loop.py and src/alignment/train_loop_kto.py:
each iteration runs self-play → SFT → self-play → KTO, so BOTH training
stages get on-policy data. Each iteration:
  1. Self-play (n_vignettes people) with the current model → SFT-format data.
  2. SFT trains a fresh LoRA on the current model using that data mixed with
     the full real (combined) corpus, then the adapter is merged.
  3. Self-play again (a fresh sample of n_vignettes people) with the
     SFT-merged model, saving every round's transcripts/critiques.
  4. build_kto_records turns those critiques into desirable/undesirable turn
     records, and run_kto trains a fresh LoRA on the SFT-merged model (the
     adapter-disabled KTO reference IS the SFT policy that generated the data).
  5. The KTO adapter is merged; the merged model is the next iteration's
     policy init, reference, and generation model.

Artifacts per iteration:
  self_play/iter_01/sft/transcripts.jsonl, critiques.jsonl
  self_play/iter_01/sft/sft_data.jsonl, sft_train.jsonl, sft_val.jsonl
  self_play/iter_01/sft/train.jsonl, val.jsonl    (self-play + real, what SFT trains on)
  self_play/iter_01/kto/transcripts.jsonl, critiques.jsonl
  self_play/iter_01/kto/transcripts_all.jsonl, critiques_all.jsonl
  self_play/iter_01/kto/kto_data.jsonl, kto_train.jsonl, kto_val.jsonl
  models_dir/Qwen3.5-4B-SFT-KTO-CACTUS-iter_01-sft/         (SFT LoRA adapter)
  models_dir/Qwen3.5-4B-SFT-KTO-CACTUS-iter_01-sftmerged/   (after SFT, full bf16)
  models_dir/Qwen3.5-4B-SFT-KTO-CACTUS-iter_01-kto/         (KTO LoRA adapter)
  models_dir/Qwen3.5-4B-SFT-KTO-CACTUS-iter_01-merged/      (after KTO, full bf16)

Usage:
    python -m src.alignment.train_loop_sft_kto --n-iters 2
    python -m src.alignment.train_loop_sft_kto --n-iters 2 --start-from-adapter
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import os
import random
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.alignment.kto import build_kto_records, merge_adapter, run_kto, MARGIN_K
from src.alignment.self_play import run_self_play
from src.alignment.sft import run_sft


# ── Defaults ──────────────────────────────────────────────────────────────────
BASE_MODEL_NAME = "Qwen/Qwen3.5-4B"
SSD_ROOT = Path(os.environ.get("SSD_ROOT", "/tmp"))

DEFAULT_MODELS_DIR    = SSD_ROOT / "models"
DEFAULT_SELF_PLAY_DIR = Path("data/self_play_sft_kto")
DEFAULT_TRAIN_DATA    = Path("data/processed/combined_train_chunked.jsonl")
DEFAULT_VAL_DATA      = Path("data/processed/combined_val_chunked.jsonl")
DEFAULT_MODEL_PREFIX  = "Qwen3.5-4B-SFT-KTO-CACTUS"

N_ITERS             = 2
N_VIGNETTES         = 300
N_REFINEMENT_CYCLES = 2
SFT_EPOCHS          = 3
KTO_EPOCHS          = 1
API_CONCURRENCY     = 500
THERAPIST_BATCH_SIZE = 8


def run_loop(
    n_iters: int = N_ITERS,
    start_iter: int = 1,
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
    kto_epochs: int = KTO_EPOCHS,
    margin_k: float = MARGIN_K,
    api_concurrency: int = API_CONCURRENCY,
    therapist_batch_size: int = THERAPIST_BATCH_SIZE,
):
    """Run the full self-play → SFT → KTO loop."""
    models_dir = Path(models_dir)
    self_play_dir = Path(self_play_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    self_play_dir.mkdir(parents=True, exist_ok=True)

    # ── Determine starting model ──────────────────────────────────────────────
    # `current_model` is always a full model (merged dir after iter 1, the raw
    # base before it). Each iteration trains fresh LoRAs on it: SFT first, then
    # KTO on the SFT-merged model, so KTO's adapter-disabled reference is this
    # iteration's SFT policy.
    if start_from_base:
        current_model = base_model_name
        print(f"[Loop-SFT-KTO] Starting from BASE model: {base_model_name}")
    else:
        # `iter_??-merged` (exactly two digits) excludes the -sftmerged dirs.
        existing = sorted(models_dir.glob(f"{model_prefix}-iter_??-merged"))
        current_model = str(existing[-1]) if existing else base_model_name
        print(f"[Loop-SFT-KTO] Starting from model: {current_model}")

    # ── Main loop ─────────────────────────────────────────────────────────────
    for iteration in range(start_iter, n_iters + 1):
        print(f"\n{'#'*70}")
        print(f"# SFT+KTO ITERATION {iteration}/{n_iters}")
        print(f"# Current model (policy init + reference + generation): {current_model}")
        print(f"{'#'*70}\n")

        # ── Step 1: Self-play for SFT data (current model, pre-SFT) ──────────
        sp_dir = self_play_dir / f"iter_{iteration:02d}" / "sft"
        sp_dir.mkdir(parents=True, exist_ok=True)

        sp_sft_data    = sp_dir / "sft_data.jsonl"     # final versions (SFT source)
        sp_transcripts = sp_dir / "transcripts.jsonl"  # final version only
        sp_critiques   = sp_dir / "critiques.jsonl"    # final version only

        print(f"\n[Loop-SFT-KTO] Step 1: Self-play (SFT data) → {sp_dir}")
        run_self_play(
            adapter_path=None,
            output_jsonl=sp_sft_data,
            output_transcripts=sp_transcripts,
            output_critiques=sp_critiques,
            base_model_name=current_model,
            n_vignettes=n_vignettes,
            vignette_seed=42 + iteration,
            n_refinement_cycles=n_refinement_cycles,
            api_concurrency=api_concurrency,
            therapist_batch_size=therapist_batch_size,
        )

        # ── Step 2: SFT on self-play data + the full real (combined) datasets ──
        # AMIE-style recipe: the full real corpus anchors EVERY iteration, with
        # the iteration's self-play dialogues layered on top of it.
        with open(sp_sft_data, encoding="utf-8") as f:
            sp_lines = [l.strip() for l in f if l.strip()]
        random.seed(42 + iteration)
        random.shuffle(sp_lines)
        split_idx = max(1, int(len(sp_lines) * 0.9))
        sp_train_lines = sp_lines[:split_idx]
        sp_val_lines = sp_lines[split_idx:]

        # Self-play-only artifacts (kept for inspection / debugging)
        (sp_dir / "sft_train.jsonl").write_text(
            "\n".join(sp_train_lines) + "\n", encoding="utf-8")
        (sp_dir / "sft_val.jsonl").write_text(
            "\n".join(sp_val_lines) + "\n", encoding="utf-8")

        with open(train_data, encoding="utf-8") as f:
            real_train_lines = [l.strip() for l in f if l.strip()]
        with open(val_data, encoding="utf-8") as f:
            real_val_lines = [l.strip() for l in f if l.strip()]

        merged_train_lines = sp_train_lines + real_train_lines
        merged_val_lines = sp_val_lines + real_val_lines
        random.shuffle(merged_train_lines)
        random.shuffle(merged_val_lines)

        train_path = sp_dir / "train.jsonl"
        val_path = sp_dir / "val.jsonl"
        train_path.write_text("\n".join(merged_train_lines) + "\n", encoding="utf-8")
        val_path.write_text("\n".join(merged_val_lines) + "\n", encoding="utf-8")

        sft_output_dir = models_dir / f"{model_prefix}-iter_{iteration:02d}-sft"
        print(f"\n[Loop-SFT-KTO] Step 2: SFT training → {sft_output_dir}")
        print(f"  Train: {len(merged_train_lines)} examples "
              f"({len(sp_train_lines)} self-play + {len(real_train_lines)} real)")
        print(f"  Val:   {len(merged_val_lines)} examples "
              f"({len(sp_val_lines)} self-play + {len(real_val_lines)} real)")

        sft_adapter_path = run_sft(
            train_data=train_path,
            val_data=val_path,
            output_dir=sft_output_dir,
            model_name=current_model,
            num_epochs=sft_epochs,
        )

        # ── Step 3: Merge SFT adapter → KTO's policy init AND reference ──────
        sft_merged_dir = models_dir / f"{model_prefix}-iter_{iteration:02d}-sftmerged"
        merge_adapter(current_model, sft_adapter_path, sft_merged_dir)
        current_model = str(sft_merged_dir)

        # ── Step 4: Self-play for KTO data (SFT-merged model, fresh sample) ──
        # On-policy for KTO: the generating model IS the KTO policy init and
        # reference. A different seed samples a fresh set of CACTUS people.
        kto_sp_dir = self_play_dir / f"iter_{iteration:02d}" / "kto"
        kto_sp_dir.mkdir(parents=True, exist_ok=True)

        kto_sp_sft_data    = kto_sp_dir / "sft_data.jsonl"     # produced, unused here
        kto_sp_transcripts = kto_sp_dir / "transcripts.jsonl"  # final version only
        kto_sp_critiques   = kto_sp_dir / "critiques.jsonl"    # final version only
        sp_rounds_t = kto_sp_dir / "transcripts_all.jsonl"  # every round (KTO source)
        sp_rounds_c = kto_sp_dir / "critiques_all.jsonl"    # every round, aligned

        print(f"\n[Loop-SFT-KTO] Step 4: Self-play (KTO data) → {kto_sp_dir}")
        run_self_play(
            adapter_path=None,
            output_jsonl=kto_sp_sft_data,
            output_transcripts=kto_sp_transcripts,
            output_critiques=kto_sp_critiques,
            output_rounds_transcripts=sp_rounds_t,
            output_rounds_critiques=sp_rounds_c,
            base_model_name=current_model,
            n_vignettes=n_vignettes,
            vignette_seed=1042 + iteration,
            n_refinement_cycles=n_refinement_cycles,
            api_concurrency=api_concurrency,
            therapist_batch_size=therapist_batch_size,
        )

        # ── Step 5: Build KTO data from EVERY round (initial+refinements+final) ─
        print(f"\n[Loop-SFT-KTO] Step 5: Building KTO records from all rounds")
        records = build_kto_records(
            sp_rounds_t, sp_rounds_c,
            margin_k=margin_k,
        )
        kto_data_path = kto_sp_dir / "kto_data.jsonl"
        with open(kto_data_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"[Loop-SFT-KTO] KTO data → {kto_data_path} ({len(records)} records)")

        # Split 90/10 into train/val (stratified so val has both labels).
        random.seed(42 + iteration)
        desir = [r for r in records if r["label"]]
        undesir = [r for r in records if not r["label"]]
        random.shuffle(desir)
        random.shuffle(undesir)

        def split(lst):
            k = max(1, int(len(lst) * 0.9)) if len(lst) > 1 else len(lst)
            return lst[:k], lst[k:]

        d_tr, d_va = split(desir)
        u_tr, u_va = split(undesir)
        train_recs = d_tr + u_tr
        val_recs = d_va + u_va
        random.shuffle(train_recs)
        random.shuffle(val_recs)

        kto_train_path = kto_sp_dir / "kto_train.jsonl"
        kto_val_path = kto_sp_dir / "kto_val.jsonl"
        kto_train_path.write_text(
            "".join(json.dumps(r) + "\n" for r in train_recs), encoding="utf-8"
        )
        kto_val_path.write_text(
            "".join(json.dumps(r) + "\n" for r in val_recs), encoding="utf-8"
        )
        print(f"[Loop-SFT-KTO] KTO Train: {len(train_recs)}  Val: {len(val_recs)}")

        # ── Step 6: KTO training on the SFT-merged model ──────────────────────
        kto_output_dir = models_dir / f"{model_prefix}-iter_{iteration:02d}-kto"
        print(f"\n[Loop-SFT-KTO] Step 6: KTO training → {kto_output_dir}")
        kto_adapter_path = run_kto(
            train_data=kto_train_path,
            val_data=kto_val_path,
            output_dir=kto_output_dir,
            model_name=current_model,
            num_epochs=kto_epochs,
        )

        # ── Step 7: Merge KTO adapter → next iteration's base AND reference ──
        merged_dir = models_dir / f"{model_prefix}-iter_{iteration:02d}-merged"
        merge_adapter(current_model, kto_adapter_path, merged_dir)
        current_model = str(merged_dir)
        print(f"\n[Loop-SFT-KTO] Iteration {iteration} complete. Model: {current_model}")

    print(f"\n{'='*70}")
    print(f"SFT+KTO TRAINING LOOP COMPLETE — {n_iters} iterations")
    print(f"Final model: {current_model}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-play → SFT → KTO training loop")
    parser.add_argument("--n-iters", type=int, default=N_ITERS)
    parser.add_argument("--start-iter", type=int, default=1,
                        help="Resume the loop at this iteration (skip completed ones)")
    parser.add_argument("--start-from-adapter", action="store_true",
                        help="Generate self-play with the latest merged model instead of base")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--model-prefix", default=DEFAULT_MODEL_PREFIX)
    parser.add_argument("--self-play-dir", type=Path, default=DEFAULT_SELF_PLAY_DIR)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA,
                        help="Path to base training data JSONL (mixed into SFT)")
    parser.add_argument("--val-data", type=Path, default=DEFAULT_VAL_DATA,
                        help="Path to validation data JSONL (mixed into SFT)")
    parser.add_argument("--base-model", default=BASE_MODEL_NAME)
    parser.add_argument("--n-vignettes", type=int, default=N_VIGNETTES)
    parser.add_argument("--n-refinement-cycles", type=int, default=N_REFINEMENT_CYCLES)
    parser.add_argument("--sft-epochs", type=int, default=SFT_EPOCHS)
    parser.add_argument("--kto-epochs", type=int, default=KTO_EPOCHS)
    parser.add_argument("--margin-k", type=float, default=MARGIN_K)
    parser.add_argument("--api-concurrency", type=int, default=API_CONCURRENCY)
    parser.add_argument("--therapist-batch-size", type=int, default=THERAPIST_BATCH_SIZE)
    args = parser.parse_args()

    run_loop(
        n_iters=args.n_iters,
        start_iter=args.start_iter,
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
        kto_epochs=args.kto_epochs,
        margin_k=args.margin_k,
        api_concurrency=args.api_concurrency,
        therapist_batch_size=args.therapist_batch_size,
    )
