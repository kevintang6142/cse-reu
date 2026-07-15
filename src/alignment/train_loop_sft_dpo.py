"""
Outer training loop (SFT+DPO variant): self-play → SFT → branching self-play → DPO → repeat.

Identical in structure to src/alignment/train_loop_sft_kto.py, except the
preference stage is turn-level DPO on branched-rollout pairs instead of KTO.
Each iteration:
  1. Self-play (UNCHANGED, critic-guided refinement) with the current model
     → SFT-format data.
  2. SFT trains a fresh LoRA on the current model using that data mixed with
     the full real (combined) corpus, then the adapter is merged.
  3. Branching self-play (src/alignment/self_play_dpo.py) with the SFT-merged
     model: at every therapist decision point, K candidates are sampled at high
     temperature, each rolled out to completion at normal temperature and
     scored with CTRS-R; best-vs-worst rollout candidates become a DPO pair and
     the conversation advances along the best branch.
  4. run_dpo trains a fresh LoRA on the SFT-merged model (the adapter-disabled
     DPO reference IS the SFT policy that generated the pairs).
  5. The DPO adapter is merged; the merged model is the next iteration's
     policy init, reference, and generation model.

Artifacts per iteration:
  self_play/iter_01/sft/transcripts.jsonl, critiques.jsonl
  self_play/iter_01/sft/sft_data.jsonl, sft_train.jsonl, sft_val.jsonl
  self_play/iter_01/sft/train.jsonl, val.jsonl    (self-play + real, what SFT trains on)
  self_play/iter_01/dpo/dpo_pairs.jsonl           (all turn-level pairs)
  self_play/iter_01/dpo/transcripts.jsonl         (main-line conversations)
  self_play/iter_01/dpo/decisions.jsonl           (per-decision rollout scores)
  self_play/iter_01/dpo/dpo_train.jsonl, dpo_val.jsonl
  models_dir/Qwen3.5-4B-SFT-DPO-CACTUS-iter_01-sft/         (SFT LoRA adapter)
  models_dir/Qwen3.5-4B-SFT-DPO-CACTUS-iter_01-sftmerged/   (after SFT, full bf16)
  models_dir/Qwen3.5-4B-SFT-DPO-CACTUS-iter_01-dpo/         (DPO LoRA adapter)
  models_dir/Qwen3.5-4B-SFT-DPO-CACTUS-iter_01-merged/      (after DPO, full bf16)

Usage:
    python -m src.alignment.train_loop_sft_dpo --n-iters 2
    python -m src.alignment.train_loop_sft_dpo --n-iters 2 --start-from-adapter
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

from src.alignment.dpo import run_dpo
from src.alignment.kto import merge_adapter
from src.alignment.self_play import run_self_play
from src.alignment.self_play_dpo import (
    BRANCH_TEMPERATURE,
    K_CANDIDATES,
    MIN_SCORE_MARGIN,
    N_VIGNETTES_DPO,
    ROLLOUT_TEMPERATURE,
    run_self_play_dpo,
)
from src.alignment.sft import run_sft


# ── Defaults ──────────────────────────────────────────────────────────────────
BASE_MODEL_NAME = "Qwen/Qwen3.5-4B"
SSD_ROOT = Path(os.environ.get("SSD_ROOT", "/tmp"))

DEFAULT_MODELS_DIR    = SSD_ROOT / "models"
DEFAULT_SELF_PLAY_DIR = Path("data/self_play_sft_dpo")
DEFAULT_TRAIN_DATA    = Path("data/processed/combined_train_chunked.jsonl")
DEFAULT_VAL_DATA      = Path("data/processed/combined_val_chunked.jsonl")
DEFAULT_MODEL_PREFIX  = "Qwen3.5-4B-SFT-DPO-CACTUS"

N_ITERS             = 2
N_VIGNETTES_SFT     = 300              # plain self-play (unchanged)
N_REFINEMENT_CYCLES = 2
SFT_EPOCHS          = 3
DPO_EPOCHS          = 1
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
    n_vignettes_sft: int = N_VIGNETTES_SFT,
    n_vignettes_dpo: int = N_VIGNETTES_DPO,
    n_refinement_cycles: int = N_REFINEMENT_CYCLES,
    sft_epochs: int = SFT_EPOCHS,
    dpo_epochs: int = DPO_EPOCHS,
    k_candidates: int = K_CANDIDATES,
    min_margin: int = MIN_SCORE_MARGIN,
    branch_temperature: float = BRANCH_TEMPERATURE,
    rollout_temperature: float = ROLLOUT_TEMPERATURE,
    api_concurrency: int = API_CONCURRENCY,
    therapist_batch_size: int = THERAPIST_BATCH_SIZE,
):
    """Run the full self-play → SFT → branching self-play → DPO loop."""
    models_dir = Path(models_dir)
    self_play_dir = Path(self_play_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    self_play_dir.mkdir(parents=True, exist_ok=True)

    # ── Determine starting model ──────────────────────────────────────────────
    # `current_model` is always a full model (merged dir after iter 1, the raw
    # base before it). Each iteration trains fresh LoRAs on it: SFT first, then
    # DPO on the SFT-merged model, so DPO's adapter-disabled reference is this
    # iteration's SFT policy.
    if start_from_base:
        current_model = base_model_name
        print(f"[Loop-SFT-DPO] Starting from BASE model: {base_model_name}")
    else:
        # `iter_??-merged` (exactly two digits) excludes the -sftmerged dirs.
        existing = sorted(models_dir.glob(f"{model_prefix}-iter_??-merged"))
        current_model = str(existing[-1]) if existing else base_model_name
        print(f"[Loop-SFT-DPO] Starting from model: {current_model}")

    # ── Main loop ─────────────────────────────────────────────────────────────
    for iteration in range(start_iter, n_iters + 1):
        print(f"\n{'#'*70}")
        print(f"# SFT+DPO ITERATION {iteration}/{n_iters}")
        print(f"# Current model (policy init + reference + generation): {current_model}")
        print(f"{'#'*70}\n")

        # ── Step 1: Self-play for SFT data (current model, pre-SFT) ──────────
        sp_dir = self_play_dir / f"iter_{iteration:02d}" / "sft"
        sp_dir.mkdir(parents=True, exist_ok=True)

        sp_sft_data    = sp_dir / "sft_data.jsonl"     # final versions (SFT source)
        sp_transcripts = sp_dir / "transcripts.jsonl"  # final version only
        sp_critiques   = sp_dir / "critiques.jsonl"    # final version only

        print(f"\n[Loop-SFT-DPO] Step 1: Self-play (SFT data) → {sp_dir}")
        run_self_play(
            adapter_path=None,
            output_jsonl=sp_sft_data,
            output_transcripts=sp_transcripts,
            output_critiques=sp_critiques,
            base_model_name=current_model,
            n_vignettes=n_vignettes_sft,
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
        print(f"\n[Loop-SFT-DPO] Step 2: SFT training → {sft_output_dir}")
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

        # ── Step 3: Merge SFT adapter → DPO's policy init AND reference ──────
        sft_merged_dir = models_dir / f"{model_prefix}-iter_{iteration:02d}-sftmerged"
        merge_adapter(current_model, sft_adapter_path, sft_merged_dir)
        current_model = str(sft_merged_dir)

        # ── Step 4: Branching self-play for DPO pairs (SFT-merged model) ─────
        # On-policy for DPO: the generating model IS the DPO policy init and
        # reference. A different seed samples a fresh set of CACTUS people.
        dpo_sp_dir = self_play_dir / f"iter_{iteration:02d}" / "dpo"
        dpo_sp_dir.mkdir(parents=True, exist_ok=True)

        dpo_pairs_path = dpo_sp_dir / "dpo_pairs.jsonl"
        print(f"\n[Loop-SFT-DPO] Step 4: Branching self-play (DPO pairs) → {dpo_sp_dir}")
        run_self_play_dpo(
            output_pairs=dpo_pairs_path,
            output_transcripts=dpo_sp_dir / "transcripts.jsonl",
            output_decisions=dpo_sp_dir / "decisions.jsonl",
            base_model_name=current_model,
            n_vignettes=n_vignettes_dpo,
            vignette_seed=1042 + iteration,
            k_candidates=k_candidates,
            min_margin=min_margin,
            branch_temperature=branch_temperature,
            rollout_temperature=rollout_temperature,
            api_concurrency=api_concurrency,
            therapist_batch_size=therapist_batch_size,
        )

        # ── Step 5: Split pairs 90/10 into train/val ──────────────────────────
        with open(dpo_pairs_path, encoding="utf-8") as f:
            pair_lines = [l.strip() for l in f if l.strip()]
        random.seed(42 + iteration)
        random.shuffle(pair_lines)
        split_idx = max(1, int(len(pair_lines) * 0.9))
        dpo_train_path = dpo_sp_dir / "dpo_train.jsonl"
        dpo_val_path = dpo_sp_dir / "dpo_val.jsonl"
        dpo_train_path.write_text(
            "\n".join(pair_lines[:split_idx]) + "\n", encoding="utf-8")
        dpo_val_path.write_text(
            "\n".join(pair_lines[split_idx:]) + "\n", encoding="utf-8")
        print(f"[Loop-SFT-DPO] DPO Train: {split_idx}  Val: {len(pair_lines) - split_idx}")

        # ── Step 6: DPO training on the SFT-merged model ──────────────────────
        dpo_output_dir = models_dir / f"{model_prefix}-iter_{iteration:02d}-dpo"
        print(f"\n[Loop-SFT-DPO] Step 6: DPO training → {dpo_output_dir}")
        dpo_adapter_path = run_dpo(
            train_data=dpo_train_path,
            val_data=dpo_val_path,
            output_dir=dpo_output_dir,
            model_name=current_model,
            num_epochs=dpo_epochs,
        )

        # ── Step 7: Merge DPO adapter → next iteration's base AND reference ──
        merged_dir = models_dir / f"{model_prefix}-iter_{iteration:02d}-merged"
        merge_adapter(current_model, dpo_adapter_path, merged_dir)
        current_model = str(merged_dir)
        print(f"\n[Loop-SFT-DPO] Iteration {iteration} complete. Model: {current_model}")

    print(f"\n{'='*70}")
    print(f"SFT+DPO TRAINING LOOP COMPLETE — {n_iters} iterations")
    print(f"Final model: {current_model}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-play → SFT → branching self-play → DPO loop")
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
    parser.add_argument("--n-vignettes-sft", type=int, default=N_VIGNETTES_SFT,
                        help="Vignettes for the plain (SFT) self-play phase")
    parser.add_argument("--n-vignettes-dpo", type=int, default=N_VIGNETTES_DPO,
                        help="Vignettes for the branching (DPO) self-play phase")
    parser.add_argument("--n-refinement-cycles", type=int, default=N_REFINEMENT_CYCLES)
    parser.add_argument("--sft-epochs", type=int, default=SFT_EPOCHS)
    parser.add_argument("--dpo-epochs", type=int, default=DPO_EPOCHS)
    parser.add_argument("--k-candidates", type=int, default=K_CANDIDATES)
    parser.add_argument("--min-margin", type=int, default=MIN_SCORE_MARGIN)
    parser.add_argument("--branch-temperature", type=float, default=BRANCH_TEMPERATURE)
    parser.add_argument("--rollout-temperature", type=float, default=ROLLOUT_TEMPERATURE)
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
        n_vignettes_sft=args.n_vignettes_sft,
        n_vignettes_dpo=args.n_vignettes_dpo,
        n_refinement_cycles=args.n_refinement_cycles,
        sft_epochs=args.sft_epochs,
        dpo_epochs=args.dpo_epochs,
        k_candidates=args.k_candidates,
        min_margin=args.min_margin,
        branch_temperature=args.branch_temperature,
        rollout_temperature=args.rollout_temperature,
        api_concurrency=args.api_concurrency,
        therapist_batch_size=args.therapist_batch_size,
    )
