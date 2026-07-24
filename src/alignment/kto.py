"""
KTO (Kahneman-Tversky Optimization) with Qwen + LoRA.

Two responsibilities:
  1. build_kto_records(...): turn self-play transcripts + CTRS-R critiques into
     KTO training records ({prompt, completion, label}) using the critic score as
     the reward signal. No expert-annotated data required.
  2. run_kto(...): QLoRA KTO training, mirroring src/alignment/sft.py.

Labelling logic (session-level reward → per-turn records):
  - Total CTRS-R score per dialogue = sum of the 11 item scores (0-33).
  - Population-percentile thresholds: dialogues in the top quantile are DESIRABLE,
    bottom quantile UNDESIRABLE, the middle is dropped (ambiguous / low-signal).
  - A dialogue with repeated therapist turns (degeneration) is forced UNDESIRABLE
    regardless of score.
  - Each dialogue explodes into one record per therapist turn: prompt = the
    conversation so far, completion = that therapist turn, label = the dialogue's
    desirable/undesirable flag.

Usage (standalone training on already-built KTO jsonl):
    python -m src.alignment.kto --train-data kto_train.jsonl --val-data kto_val.jsonl \
                                --output-dir /path/to/output
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
# KTO runs an extra full-vocab KL forward pass; with Qwen's ~250k vocab the fp32
# logit cast is huge, so guard against fragmentation-driven OOM.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl.experimental.kto import KTOConfig, KTOTrainer

# The therapist system prompt the self-play dialogues were generated under.
# Reused so KTO prompts match what the model will actually see at inference.
from src.alignment.self_play import THERAPIST_SYSTEM, CTSR_ITEMS


# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_NAME     = "Qwen/Qwen3.5-4B"
LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
NUM_EPOCHS     = 1
# KTO does a main forward + an extra KL forward, each producing full-vocab logits
# (Qwen vocab ≈ 250k), and its KL term REQUIRES an actual batch > 1. So batch stays
# at the floor of 2, and batch×seq drives the fp32 logit-cast peak on the 48 GB card.
# max_length matches self_play.MAX_SEQ_LENGTH (4096) so KTO prompts keep the system
# prompt + full history the SFT chunks trained with, instead of left-truncating ~30%
# of records at 2048.
BATCH_SIZE     = 2
GRAD_ACCUM     = 8
LR             = 5e-6     # much lower than SFT
BETA           = 0.1
MAX_SEQ_LENGTH = 4096     # left-truncates prompt; therapist completion is preserved

# Labelling default: margin in units of the score distribution's std-dev.
# good if score >= mean + k·σ, bad if score <= mean − k·σ, else dropped. Ties the
# cutoff to the spread of the batch, so a tight (uniformly-similar) batch yields few
# labels instead of forcing a fixed fraction to be "bad".
MARGIN_K       = 0.5
N_ITEMS        = len(CTSR_ITEMS)  # 11

USER_START_MSG = {"role": "user", "content": "Start the session."}


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ══════════════════════════════════════════════════════════════════════════════
# Scoring + labelling
# ══════════════════════════════════════════════════════════════════════════════

def total_ctsr_score(critique: dict) -> int | None:
    """Sum the 11 CTRS-R item scores. Returns None if the critique failed to parse
    or is missing any item (so we never train on a partially-scored dialogue)."""
    parsed = critique.get("parsed")
    if not isinstance(parsed, dict):
        return None
    total, n = 0, 0
    for v in parsed.values():
        if isinstance(v, dict) and isinstance(v.get("score"), (int, float)):
            total += int(v["score"])
            n += 1
    return total if n == N_ITEMS else None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def is_degenerate(turns: list[dict], sim_threshold: float = 0.9) -> bool:
    """True if two therapist turns are verbatim or near-verbatim duplicates —
    the loop-collapse failure mode. Such dialogues are forced UNDESIRABLE."""
    ther = [t["content"] for t in turns if t["role"] == "therapist"]
    seen = [_norm(x) for x in ther]
    for i in range(len(seen)):
        for j in range(i + 1, len(seen)):
            if seen[i] == seen[j]:
                return True
            if SequenceMatcher(None, seen[i], seen[j]).ratio() >= sim_threshold:
                return True
    return False


def _turn_records(turns: list[dict], label: bool) -> list[dict]:
    """Explode one dialogue into per-therapist-turn KTO records.

    prompt     = system + conversation so far (conversational format)
    completion = the single therapist (assistant) turn
    label      = dialogue-level desirable/undesirable flag
    """
    records = []
    history = [{"role": "system", "content": THERAPIST_SYSTEM}]
    for turn in turns:
        if turn["role"] == "therapist":
            prompt = list(history)
            # Guarantee a user turn exists before the assistant completion
            # (the therapist speaks first, so the opening turn has none).
            if not any(m["role"] == "user" for m in prompt):
                prompt = [prompt[0], USER_START_MSG] + prompt[1:]
            records.append({
                "prompt": prompt,
                "completion": [{"role": "assistant", "content": turn["content"]}],
                "label": bool(label),
                # Self-play generates with thinking disabled (see self_play.py), so
                # tokenize KTO the same way. This also keeps the prompt a clean prefix
                # of prompt+completion — otherwise Qwen3's empty <think> scaffold
                # differs between the two renders and TRL warns on every record.
                "chat_template_kwargs": {"enable_thinking": False},
            })
            history.append({"role": "assistant", "content": turn["content"]})
        else:
            history.append({"role": "user", "content": turn["content"]})
    return records


def _load_scored(transcripts_path, critiques_path) -> list[tuple[dict, int | None]]:
    """Load index-aligned (transcript, total_score) pairs."""
    transcripts = load_jsonl(transcripts_path)
    critiques = load_jsonl(critiques_path)
    if len(transcripts) != len(critiques):
        raise ValueError(
            f"transcripts ({len(transcripts)}) and critiques ({len(critiques)}) "
            "are not the same length — cannot align by index."
        )
    scored = []
    for t, c in zip(transcripts, critiques):
        if t["vignette"] != c["vignette"]:
            raise ValueError(
                f"Index misalignment: transcript={t['vignette']} critique={c['vignette']}"
            )
        scored.append((t, total_ctsr_score(c)))
    return scored


def build_kto_records(
    transcripts_path: str | Path,
    critiques_path: str | Path,
    margin_k: float = MARGIN_K,
) -> list[dict]:
    """Build KTO records from self-play transcripts + CTRS-R critiques.

    Labels by a std-dev margin around the mean:
        desirable   if score >= mean + margin_k·σ
        undesirable if score <= mean − margin_k·σ
        dropped     otherwise
    Degenerate (repetition) dialogues are forced undesirable; unparseable dropped.
    Transcripts and critiques must be index-aligned (as written by run_self_play).
    """
    scored = _load_scored(transcripts_path, critiques_path)
    valid_scores = np.array([s for _, s in scored if s is not None], dtype=float)
    if len(valid_scores) == 0:
        raise ValueError("No parseable critiques — cannot threshold.")

    mu = float(valid_scores.mean())
    sd = float(valid_scores.std())
    good_thr = mu + margin_k * sd
    bad_thr = mu - margin_k * sd

    records: list[dict] = []
    n_good = n_bad = n_drop = n_skip = n_degen = 0
    for t, score in scored:
        if score is None:
            n_skip += 1
            continue
        degen = is_degenerate(t["turns"])
        if degen:
            label = False                 # force degenerate dialogues to undesirable
            n_degen += 1
        elif score >= good_thr:
            label = True
        elif score <= bad_thr:
            label = False
        else:
            n_drop += 1                    # within ±k·σ of the mean — too ambiguous
            continue
        records.extend(_turn_records(t["turns"], label))
        n_good += int(label)
        n_bad += int(not label)

    n_desir = sum(r["label"] for r in records)
    n_undesir = len(records) - n_desir
    print(
        f"[KTO] Labelling: mean={mu:.1f} σ={sd:.1f} k={margin_k} → "
        f"good_thr(≥{good_thr:.1f}) bad_thr(≤{bad_thr:.1f})\n"
        f"[KTO] Dialogues: desirable={n_good} undesirable={n_bad} "
        f"(forced-by-repetition={n_degen}) dropped-middle={n_drop} "
        f"parse-failed={n_skip}\n"
        f"[KTO] Turn records: desirable={n_desir} undesirable={n_undesir} "
        f"total={len(records)}"
    )
    return records


def sweep_labels(transcripts_path, critiques_path, ks=(0.25, 0.5, 0.75, 1.0)) -> None:
    """Print how many good/bad dialogues each margin_k yields — a sensitivity check
    so the cutoff choice is empirical rather than arbitrary."""
    scored = _load_scored(transcripts_path, critiques_path)
    s = np.array([x for _, x in scored if x is not None], dtype=float)
    mu, sd = float(s.mean()), float(s.std())
    print(f"[sweep] n_scored={len(s)} mean={mu:.1f} σ={sd:.1f} "
          f"(min={s.min():.0f} max={s.max():.0f})")
    for k in ks:
        good = int((s >= mu + k * sd).sum())
        bad = int((s <= mu - k * sd).sum())
        drop = len(s) - good - bad
        print(f"[sweep] k={k:<4}  good_thr≥{mu+k*sd:4.1f}  bad_thr≤{mu-k*sd:4.1f}  "
              f"→ good={good:3d}  bad={bad:3d}  drop={drop:3d}")


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def run_kto(
    train_data: str | Path,
    val_data: str | Path,
    output_dir: str | Path,
    model_name: str = MODEL_NAME,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    grad_accum: int = GRAD_ACCUM,
    lr: float = LR,
    beta: float = BETA,
    max_seq_length: int = MAX_SEQ_LENGTH,
    desirable_weight: float | None = None,
    undesirable_weight: float | None = None,
) -> Path:
    """Run KTO training and save LoRA adapters. Returns the adapter directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[KTO] Model     : {model_name}")
    print(f"[KTO] Train     : {train_data}")
    print(f"[KTO] Val       : {val_data}")
    print(f"[KTO] Output    : {output_dir}")

    train_records = load_jsonl(train_data)
    val_records = load_jsonl(val_data)
    train_dataset = Dataset.from_list(train_records)
    val_dataset = Dataset.from_list(val_records)

    # ── Auto-balance classes via label weights ────────────────────────────────
    n_desir = sum(bool(r["label"]) for r in train_records)
    n_undesir = len(train_records) - n_desir
    if desirable_weight is None:
        desirable_weight = 1.0
    if undesirable_weight is None:
        # Make desirable_weight*n_desir ≈ undesirable_weight*n_undesir (ratio ~1).
        undesirable_weight = (n_desir / n_undesir) if n_undesir > 0 else 1.0
    print(f"[KTO] Train records: desirable={n_desir} undesirable={n_undesir}")
    print(f"[KTO] Weights      : desirable={desirable_weight:.3f} undesirable={undesirable_weight:.3f}")
    if n_desir == 0 or n_undesir == 0:
        raise ValueError("KTO needs both desirable and undesirable examples in train.")

    # ── Tokenizer & model (QLoRA 4-bit) ───────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # 10% warmup, expressed in steps (warmup_ratio is deprecated in transformers v5)
    effective_batch = batch_size * grad_accum
    steps_per_epoch = (len(train_records) + effective_batch - 1) // effective_batch
    warmup_steps = max(1, int(0.1 * steps_per_epoch * num_epochs))

    kto_config = KTOConfig(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        beta=beta,
        desirable_weight=desirable_weight,
        undesirable_weight=undesirable_weight,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=10,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=20,
        bf16=use_bf16,
        fp16=(not use_bf16) and torch.cuda.is_available(),
        gradient_checkpointing=True,
        report_to="none",
        max_length=max_seq_length,
        remove_unused_columns=False,  # required by KTO's unpaired collator
    )

    # ref_model=None + peft_config: KTO uses the adapter-disabled base as reference,
    # so there is no separate reference-model VRAM cost.
    trainer = KTOTrainer(
        model=model,
        ref_model=None,
        args=kto_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print(f"[KTO] Steps/epoch: {len(trainer.get_train_dataloader())}")
    result = trainer.train()
    print(f"[KTO] Training complete — loss: {result.metrics.get('train_loss', float('nan')):.4f}")

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[KTO] Adapters saved → {output_dir}")

    del model, trainer
    torch.cuda.empty_cache()
    return output_dir


def merge_adapter(
    base_model_name: str,
    adapter_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Merge a LoRA adapter into its base model and save the result (bf16).

    Used for canonical iterated preference training: the merged model becomes
    the NEXT iteration's base, so both the policy init and the adapter-disabled
    reference are the previous iteration's policy. Merging is done on CPU in
    bf16 — the adapter was trained against the 4-bit quantized base, so this is
    the standard (slightly lossy) QLoRA merge.
    """
    from peft import PeftModel

    output_dir = Path(output_dir)
    print(f"[KTO] Merging {adapter_path} into {base_model_name} → {output_dir}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model = model.merge_and_unload()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    del model
    print(f"[KTO] Merged model saved → {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KTO with LoRA")
    parser.add_argument("--train-data", required=True, help="KTO train JSONL ({prompt,completion,label})")
    parser.add_argument("--val-data", required=True, help="KTO val JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--beta", type=float, default=BETA)
    args = parser.parse_args()

    run_kto(
        train_data=args.train_data,
        val_data=args.val_data,
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        beta=args.beta,
    )
