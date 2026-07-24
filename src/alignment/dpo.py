"""
DPO (Direct Preference Optimization) with Qwen + LoRA.

Trains on turn-level preference pairs produced by src/alignment/self_play_dpo.py:
{prompt, chosen, rejected} where prompt is the shared conversation history and
chosen/rejected are two single therapist turns branched from that exact history —
the same-context comparison the DPO objective assumes. QLoRA setup mirrors
src/alignment/kto.py.

Usage (standalone training on already-built DPO jsonl):
    python -m src.alignment.dpo --train-data dpo_train.jsonl --val-data dpo_val.jsonl \
                                --output-dir /path/to/output
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer


# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_NAME     = "Qwen/Qwen3.5-4B"
LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
NUM_EPOCHS     = 1
# DPO forwards chosen+rejected CONCATENATED (2 sequences per record) through both
# policy and reference, and accelerate casts the full-vocab logits (Qwen ~250k)
# to fp32 — batch_size 2 meant 4×~3k-token sequences and an 11GiB logit cast that
# OOMed the 48GB card. batch 1 halves the peak; grad_accum 16 keeps the same
# effective batch of 16.
BATCH_SIZE     = 1
GRAD_ACCUM     = 16
LR             = 5e-6     # much lower than SFT
BETA           = 0.1
# Matches self_play.MAX_SEQ_LENGTH (4096). Over-length prompts are trimmed
# client-side (oldest turns dropped, system prompt + recent history + completion
# kept) so TRL's sequence truncation never fires.
MAX_SEQ_LENGTH = 4096


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _strip_meta(records: list[dict]) -> list[dict]:
    """Drop bookkeeping fields self_play_dpo attaches — TRL only needs
    prompt/chosen/rejected (+ chat_template_kwargs)."""
    return [{k: v for k, v in r.items() if k != "meta"} for r in records]


def _truncate_prompts(records: list[dict], tokenizer, max_length: int) -> list[dict]:
    """Trim over-length prompts by dropping the OLDEST user+assistant exchange
    (keeping the system prompt) until prompt + longest completion fit in
    max_length. Dropping exchanges pairwise preserves role alternation. This
    keeps the completion and recent history intact, so TRL's own sequence
    truncation (which would cut the completion tail) never triggers."""
    margin = 64  # template/special-token headroom

    def n_tokens(messages):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
        return len(tokenizer.encode(text))

    n_trimmed = 0
    for r in records:
        completion = max(r["chosen"], r["rejected"], key=lambda c: len(c[0]["content"]))
        trimmed = False
        while (len(r["prompt"]) > 3
               and n_tokens(r["prompt"] + completion) + margin > max_length):
            # prompt = [system, user, assistant, user, ...] — drop the oldest
            # user+assistant exchange after the system message.
            r["prompt"] = [r["prompt"][0]] + r["prompt"][3:]
            trimmed = True
        n_trimmed += trimmed
    if n_trimmed:
        print(f"[DPO] Trimmed oldest turns from {n_trimmed}/{len(records)} over-length prompts")
    return records


def run_dpo(
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
) -> Path:
    """Run DPO training and save LoRA adapters. Returns the adapter directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DPO] Model     : {model_name}")
    print(f"[DPO] Train     : {train_data}")
    print(f"[DPO] Val       : {val_data}")
    print(f"[DPO] Output    : {output_dir}")

    train_records = _strip_meta(load_jsonl(train_data))
    val_records = _strip_meta(load_jsonl(val_data))
    print(f"[DPO] Train pairs: {len(train_records)}  Val pairs: {len(val_records)}")
    if not train_records:
        raise ValueError("No DPO training pairs.")

    # ── Tokenizer & model (QLoRA 4-bit) ───────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_records = _truncate_prompts(train_records, tokenizer, max_seq_length)
    val_records = _truncate_prompts(val_records, tokenizer, max_seq_length)
    train_dataset = Dataset.from_list(train_records)
    val_dataset = Dataset.from_list(val_records)

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

    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        beta=beta,
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
    )

    # ref_model=None + peft_config: DPO uses the adapter-disabled base as reference,
    # so there is no separate reference-model VRAM cost.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print(f"[DPO] Steps/epoch: {len(trainer.get_train_dataloader())}")
    result = trainer.train()
    print(f"[DPO] Training complete — loss: {result.metrics.get('train_loss', float('nan')):.4f}")

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[DPO] Adapters saved → {output_dir}")

    del model, trainer
    torch.cuda.empty_cache()
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DPO with LoRA")
    parser.add_argument("--train-data", required=True, help="DPO train JSONL ({prompt,chosen,rejected})")
    parser.add_argument("--val-data", required=True, help="DPO val JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--beta", type=float, default=BETA)
    args = parser.parse_args()

    run_dpo(
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
