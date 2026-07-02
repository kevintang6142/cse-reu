"""
Supervised Fine-Tuning (SFT) with Qwen + LoRA.

Callable as a module function or standalone script.
Usage:
    python -m src.scripts.sft --train-data data/processed/combined_train_chunked.jsonl \
                              --val-data data/processed/combined_val_chunked.jsonl \
                              --output-dir /path/to/output
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_NAME     = "Qwen/Qwen3.5-4B"
LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
NUM_EPOCHS     = 3
BATCH_SIZE     = 1
GRAD_ACCUM     = 8
LR             = 3e-5
MAX_SEQ_LENGTH = 4096


def load_jsonl(path: str | Path) -> list[dict]:
    """Load JSONL records."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_sft(
    train_data: str | Path,
    val_data: str | Path,
    output_dir: str | Path,
    model_name: str = MODEL_NAME,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    grad_accum: int = GRAD_ACCUM,
    lr: float = LR,
    max_seq_length: int = MAX_SEQ_LENGTH,
) -> Path:
    """
    Run SFT training and save LoRA adapters.

    Returns the path to the saved adapter directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_dir  # Save directly in output_dir, no subfolder

    print(f"[SFT] Model        : {model_name}")
    print(f"[SFT] Train data   : {train_data}")
    print(f"[SFT] Val data     : {val_data}")
    print(f"[SFT] Output       : {adapter_dir}")
    print(f"[SFT] Epochs       : {num_epochs}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    train_records = load_jsonl(train_data)
    val_records = load_jsonl(val_data)
    train_dataset = Dataset.from_list(train_records)
    val_dataset = Dataset.from_list(val_records)
    print(f"[SFT] Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # ── Load tokenizer & model ────────────────────────────────────────────────
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
    model.config.eos_token_id = tokenizer.eos_token_id
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    # ── Apply LoRA ────────────────────────────────────────────────────────────
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
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Train ─────────────────────────────────────────────────────────────────
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=10,
        per_device_eval_batch_size=batch_size,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16,
        fp16=(not use_bf16) and torch.cuda.is_available(),
        gradient_checkpointing=True,
        report_to="none",
        max_length=max_seq_length,
        dataset_kwargs={"skip_prepare_dataset": False},
        assistant_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    print(f"[SFT] Steps/epoch: {len(trainer.get_train_dataloader())}")
    train_result = trainer.train()

    print(f"[SFT] Training complete — loss: {train_result.metrics['train_loss']:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"[SFT] Adapters saved → {adapter_dir}")

    # Cleanup
    del model, trainer
    torch.cuda.empty_cache()

    return adapter_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SFT with LoRA")
    parser.add_argument("--train-data", required=True, help="Path to train JSONL")
    parser.add_argument("--val-data", required=True, help="Path to val JSONL")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()

    run_sft(
        train_data=args.train_data,
        val_data=args.val_data,
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
    )
