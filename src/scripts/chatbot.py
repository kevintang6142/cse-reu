"""
chatbot.py — Interactive comparison of base vs. fine-tuned therapist models.

You play the patient; both models respond each turn so you can compare them
side-by-side. Uses the same generation logic as notebooks/04_eval.ipynb.

Usage:
    uv run src/scripts/chatbot.py [options]

Options:
    --base   BASE_MODEL   HuggingFace model ID for the base model
                          (default: Qwen/Qwen3.5-9B)
    --sft    SFT_PATH     Path to LoRA adapters or merged fine-tuned model
                          (default: $SSD_ROOT/models/qwen3.5-sft-new/lora-adapters)
    --turns  N            Max turns per session (default: unlimited)
    --save   FILE         Save transcript to FILE as JSONL when done
"""

import argparse
import json
import os
import re
import textwrap
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv(Path(__file__).parent.parent.parent / ".env")

# ── ANSI colours ────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
BASE_C = "\033[94m"   # blue  — base model
SFT_C  = "\033[93m"   # yellow — SFT model
USER_C = "\033[92m"   # green  — patient (you)
SEP_C  = "\033[90m"   # grey   — separators


def cprint(text: str, colour: str = "", bold: bool = False) -> None:
    prefix = (BOLD if bold else "") + colour
    print(f"{prefix}{text}{RESET}")


def separator(char: str = "─", width: int = 70, colour: str = SEP_C) -> None:
    cprint(char * width, colour)


THERAPIST_SYSTEM = (
    "You are a skilled CBT (Cognitive Behavioural Therapy) therapist conducting an initial "
    "clinical assessment session. Follow CBT principles:\n"
    "- Open with a collaborative agenda-setting check-in\n"
    "- Use Socratic questioning to explore thoughts, feelings, and behaviours\n"
    "- Help the patient identify links between cognitions, emotions, and behaviours\n"
    "- Validate emotions while gently challenging unhelpful thought patterns\n"
    "- Work toward a shared conceptualisation and practical goal-oriented interventions\n"
    "- Be warm, empathic, genuine, and professionally boundaried\n"
    "- Propose appropriate between-session homework toward the end of the session\n\n"
    "Respond as a therapist — ask thoughtful questions, reflect back, guide toward insight. "
    "Keep responses focused (2–4 sentences)."
)


# ── Generation helpers (identical to 04_eval.ipynb) ─────────────────────────
def strip_thinking(text: str) -> str:
    """Strip Qwen3 <think>...</think> chain-of-thought blocks."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def to_hf_messages(turns: list[dict], system_prompt: str) -> list[dict]:
    """Convert shared turn list → HuggingFace chat format (therapist POV)."""
    messages = [{"role": "system", "content": system_prompt}]
    for t in turns:
        role = "user" if t["role"] == "patient" else "assistant"
        messages.append({"role": role, "content": t["content"]})
    return messages


def therapist_turn(model, tokenizer, turns: list[dict], max_new_tokens: int = 1200) -> str:
    """Generate the therapist's next utterance; strips CoT if present."""
    messages = to_hf_messages(turns, THERAPIST_SYSTEM)

    # Qwen3 chat template requires at least one user message
    if len(messages) == 1:
        messages.append({"role": "user", "content": "Please begin the session."})

    try:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Ensure generation stops at <|im_end|> (end of assistant turn).
    # convert_tokens_to_ids may return None for models without that token; filter it out.
    stop_ids = [
        i for i in {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
        if isinstance(i, int) and i >= 0
    ]

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            eos_token_id=stop_ids,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return strip_thinking(raw)


# ── Model loading (identical to 04_eval.ipynb) ───────────────────────────────
def load_model(model_path: str, base_model_name: str | None = None):
    """
    Load a model for inference.

    If model_path contains adapter_config.json it is treated as a PEFT/LoRA
    adapter and merged onto base_model_name. Otherwise it is loaded directly.
    """
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    path = Path(model_path)
    is_peft = (path / "adapter_config.json").exists()

    if is_peft:
        from peft import PeftModel  # type: ignore
        if not base_model_name:
            raise ValueError("--base required when loading a PEFT adapter")
        print("  Detected PEFT adapter — loading base + adapter and merging…")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
    else:
        print(f"  Loading model from {model_path}…")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Loaded {n_params:.1f}B parameters.")
    return model, tokenizer


# ── CLI ───────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    ssd_root = os.environ.get("SSD_ROOT", "/tmp")
    default_sft = str(Path(ssd_root) / "models/Qwen3.5-4B-SFT/lora-adapters")

    p = argparse.ArgumentParser(
        description="Interactive base-vs-SFT therapist chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--base",     default="Qwen/Qwen3.5-4B",  help="Base model ID or path")
    p.add_argument("--sft",      default=default_sft,         help="SFT model/adapter path")
    p.add_argument("--turns",    type=int, default=0,
                   help="Max turns (0 = unlimited)")
    p.add_argument("--save",     default=None, metavar="FILE",
                   help="Save transcript to JSONL file when done")
    return p


# ── UI helpers ────────────────────────────────────────────────────────────────
def wrap(text: str, width: int = 68, indent: str = "  ") -> str:
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def print_model_turn(label: str, colour: str, text: str) -> None:
    separator()
    cprint(f" {label}", colour, bold=True)
    separator()
    print(wrap(text))
    print()


def print_header(base_name: str, sft_path: str) -> None:
    separator("═", 70)
    cprint("  BASE vs. SFT Therapist Chatbot", BOLD, bold=True)
    separator("═", 70)
    cprint(f"  Base model : {base_name}", BASE_C)
    cprint(f"  SFT model  : {sft_path}", SFT_C)
    separator("═", 70)
    cprint("  Type your response after each therapist turn.", DIM)
    cprint("  Commands: 'quit' / 'exit' to end, 'save' to save transcript.", DIM)
    separator("═", 70)
    print()


# ── Main loop ─────────────────────────────────────────────────────────────────
def run_chatbot(args: argparse.Namespace) -> None:
    # ── load models ──
    separator("═", 70)
    cprint("\nLoading BASE model…", BASE_C, bold=True)
    base_model, base_tok = load_model(args.base)

    separator("─")
    cprint("\nLoading SFT model…", SFT_C, bold=True)
    sft_model, sft_tok = load_model(args.sft, base_model_name=args.base)
    print()

    # ── print header ──
    print_header(args.base, args.sft)

    # ── shared turn history (each model has its own copy) ──
    base_turns: list[dict] = []
    sft_turns:  list[dict] = []

    # accumulated transcript for optional save
    transcript: list[dict] = []

    turn_num = 0

    while True:
        turn_num += 1
        if args.turns > 0 and turn_num > args.turns:
            cprint(f"\n[Reached max turns: {args.turns}]", DIM)
            break

        # ── generate both therapist responses ──
        cprint(f"  Generating turn {turn_num}…", DIM)

        base_reply = therapist_turn(base_model, base_tok, base_turns)
        sft_reply  = therapist_turn(sft_model,  sft_tok,  sft_turns)

        # ── display ──
        print_model_turn("BASE THERAPIST", BASE_C, base_reply)
        print_model_turn("SFT THERAPIST",  SFT_C,  sft_reply)

        transcript.append({"turn": turn_num, "base": base_reply, "sft": sft_reply})

        # ── patient input ──
        separator("═", 70)
        cprint("  YOU:", USER_C, bold=True)
        separator("═", 70)

        try:
            user_input = input(f"{USER_C}> {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "q"}:
            break

        if user_input.lower() == "save":
            _save_transcript(transcript, args.save or "chatbot_transcript.jsonl")
            continue

        # ── append to turn histories (therapist first, then patient) ──
        # mirrors run_interview in 04_eval.ipynb: turns ends with patient msg
        # so the next therapist_turn call sees it as the last context item.
        base_turns.append({"role": "therapist", "content": base_reply})
        base_turns.append({"role": "patient",   "content": user_input})
        sft_turns.append( {"role": "therapist", "content": sft_reply})
        sft_turns.append( {"role": "patient",   "content": user_input})

        transcript[-1]["patient"] = user_input
        print()

    # ── end of session ──
    separator("═", 70)
    cprint("  Session ended.", BOLD, bold=True)
    separator("═", 70)

    if args.save:
        _save_transcript(transcript, args.save)


def _save_transcript(transcript: list[dict], filepath: str) -> None:
    out = Path(filepath)
    record = {"turns": transcript}
    with open(out, "w") as f:
        f.write(json.dumps(record) + "\n")
    cprint(f"  Transcript saved → {out.resolve()}", DIM)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_chatbot(args)
