"""
Outer training loop (KTO variant): self-play → KTO → self-play → KTO → ...

Identical in structure to src/alignment/train_loop.py, except step 2 is KTO
instead of SFT. Each iteration:
  1. Self-play generates dialogues using the current model + CTRS-R critiques.
  2. build_kto_records turns those critiques into desirable/undesirable turn
     records (the critic score is the reward — no expert data needed).
  3. run_kto trains a fresh LoRA on the current model (canonical iterated
     preference training: the adapter-disabled reference IS the previous
     iteration's policy).
  4. The adapter is merged into the current model; the merged model is the
     next iteration's policy init, reference, and generation model.

Artifacts per iteration:
  self_play/iter_01/transcripts.jsonl
  self_play/iter_01/critiques.jsonl
  self_play/iter_01/kto_data.jsonl      (all labelled turn records)
  self_play/iter_01/kto_train.jsonl
  self_play/iter_01/kto_val.jsonl
  models_dir/Qwen3.5-4B-KTO-CACTUS-iter_01/         (LoRA adapter)
  models_dir/Qwen3.5-4B-KTO-CACTUS-iter_01-merged/  (full model, bf16)

Usage:
    python -m src.alignment.train_loop_kto --n-iters 2
    python -m src.alignment.train_loop_kto --n-iters 2 --start-from-adapter
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


# ── Defaults ──────────────────────────────────────────────────────────────────
BASE_MODEL_NAME = "Qwen/Qwen3.5-4B"
SSD_ROOT = Path(os.environ.get("SSD_ROOT", "/tmp"))

DEFAULT_MODELS_DIR    = SSD_ROOT / "models"
DEFAULT_SELF_PLAY_DIR = Path("data/self_play_kto")
DEFAULT_MODEL_PREFIX  = "Qwen3.5-4B-KTO-CACTUS"

N_ITERS             = 2
N_VIGNETTES         = 300
N_REFINEMENT_CYCLES = 2
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
    base_model_name: str = BASE_MODEL_NAME,
    n_vignettes: int = N_VIGNETTES,
    n_refinement_cycles: int = N_REFINEMENT_CYCLES,
    kto_epochs: int = KTO_EPOCHS,
    margin_k: float = MARGIN_K,
    api_concurrency: int = API_CONCURRENCY,
    therapist_batch_size: int = THERAPIST_BATCH_SIZE,
):
    """Run the full self-play → KTO loop."""
    models_dir = Path(models_dir)
    self_play_dir = Path(self_play_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    self_play_dir.mkdir(parents=True, exist_ok=True)

    # ── Determine starting model ──────────────────────────────────────────────
    # Canonical iterated preference training: each iteration trains a fresh LoRA
    # on `current_model` (so the adapter-disabled KTO reference IS the previous
    # iteration's policy), then merges it. `current_model` is a merged full model
    # dir after iter 1, the raw base before it.
    if start_from_base:
        current_model = base_model_name
        print(f"[Loop-KTO] Starting from BASE model: {base_model_name}")
    else:
        existing = sorted(models_dir.glob(f"{model_prefix}-iter_*-merged"))
        current_model = str(existing[-1]) if existing else base_model_name
        print(f"[Loop-KTO] Starting from model: {current_model}")

    # ── Main loop ─────────────────────────────────────────────────────────────
    for iteration in range(start_iter, n_iters + 1):
        print(f"\n{'#'*70}")
        print(f"# KTO ITERATION {iteration}/{n_iters}")
        print(f"# Current model (policy init + reference + generation): {current_model}")
        print(f"{'#'*70}\n")

        # ── Step 1: Self-play ─────────────────────────────────────────────────
        sp_dir = self_play_dir / f"iter_{iteration:02d}"
        sp_dir.mkdir(parents=True, exist_ok=True)

        sp_sft_data    = sp_dir / "sft_data.jsonl"     # produced by run_self_play (unused here)
        sp_transcripts = sp_dir / "transcripts.jsonl"  # final version only
        sp_critiques   = sp_dir / "critiques.jsonl"    # final version only
        sp_rounds_t    = sp_dir / "transcripts_all.jsonl"  # every round (KTO source)
        sp_rounds_c    = sp_dir / "critiques_all.jsonl"    # every round, aligned

        print(f"\n[Loop-KTO] Step 1: Self-play → {sp_dir}")
        run_self_play(
            adapter_path=None,
            output_jsonl=sp_sft_data,
            output_transcripts=sp_transcripts,
            output_critiques=sp_critiques,
            output_rounds_transcripts=sp_rounds_t,
            output_rounds_critiques=sp_rounds_c,
            base_model_name=current_model,
            n_vignettes=n_vignettes,
            vignette_seed=42 + iteration,
            n_refinement_cycles=n_refinement_cycles,
            api_concurrency=api_concurrency,
            therapist_batch_size=therapist_batch_size,
        )

        # ── Step 2: Build KTO data from EVERY round (initial+refinements+final) ─
        print(f"\n[Loop-KTO] Step 2: Building KTO records from all rounds")
        records = build_kto_records(
            sp_rounds_t, sp_rounds_c,
            margin_k=margin_k,
        )
        kto_data_path = sp_dir / "kto_data.jsonl"
        with open(kto_data_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"[Loop-KTO] KTO data → {kto_data_path} ({len(records)} records)")

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

        kto_train_path = sp_dir / "kto_train.jsonl"
        kto_val_path = sp_dir / "kto_val.jsonl"
        kto_train_path.write_text(
            "".join(json.dumps(r) + "\n" for r in train_recs), encoding="utf-8"
        )
        kto_val_path.write_text(
            "".join(json.dumps(r) + "\n" for r in val_recs), encoding="utf-8"
        )
        print(f"[Loop-KTO] Train: {len(train_recs)}  Val: {len(val_recs)}")

        # ── Step 3: KTO training ──────────────────────────────────────────────
        iter_output_dir = models_dir / f"{model_prefix}-iter_{iteration:02d}"
        print(f"\n[Loop-KTO] Step 3: KTO training → {iter_output_dir}")
        adapter_path = run_kto(
            train_data=kto_train_path,
            val_data=kto_val_path,
            output_dir=iter_output_dir,
            model_name=current_model,
            num_epochs=kto_epochs,
        )

        # ── Step 4: Merge adapter → next iteration's base AND reference ──────
        merged_dir = models_dir / f"{model_prefix}-iter_{iteration:02d}-merged"
        merge_adapter(current_model, adapter_path, merged_dir)
        current_model = str(merged_dir)
        print(f"\n[Loop-KTO] Iteration {iteration} complete. Model: {current_model}")

    print(f"\n{'='*70}")
    print(f"KTO TRAINING LOOP COMPLETE — {n_iters} iterations")
    print(f"Final model: {current_model}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-play → KTO training loop")
    parser.add_argument("--n-iters", type=int, default=N_ITERS)
    parser.add_argument("--start-iter", type=int, default=1,
                        help="Resume the loop at this iteration (skip completed ones)")
    parser.add_argument("--start-from-adapter", action="store_true",
                        help="Generate self-play with an existing adapter instead of base")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--model-prefix", default=DEFAULT_MODEL_PREFIX)
    parser.add_argument("--self-play-dir", type=Path, default=DEFAULT_SELF_PLAY_DIR)
    parser.add_argument("--base-model", default=BASE_MODEL_NAME)
    parser.add_argument("--n-vignettes", type=int, default=N_VIGNETTES)
    parser.add_argument("--n-refinement-cycles", type=int, default=N_REFINEMENT_CYCLES)
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
        base_model_name=args.base_model,
        n_vignettes=args.n_vignettes,
        n_refinement_cycles=args.n_refinement_cycles,
        kto_epochs=args.kto_epochs,
        margin_k=args.margin_k,
        api_concurrency=args.api_concurrency,
        therapist_batch_size=args.therapist_batch_size,
    )
