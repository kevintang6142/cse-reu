"""
compare_ui.py — Side-by-side comparison of two therapist models on a single response.

Build a chat history (therapist + patient turns), then generate the next
therapist reply from two models simultaneously for comparison.

Usage:
    uv run streamlit run src/scripts/compare_ui.py
"""

import gc
import json
import os
import re
from pathlib import Path

import streamlit as st
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv(Path(__file__).parent.parent.parent / ".env")

SSD_ROOT = Path(os.environ.get("SSD_ROOT", "/tmp"))
MODELS_DIR = SSD_ROOT / "models"
BASE_MODEL_ID = "Qwen/Qwen3.5-4B"

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


# ── Helpers ────────────────────────────────────────────────────────────────────

def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def discover_models() -> dict[str, str]:
    models = {"Base (Qwen3.5-4B)": BASE_MODEL_ID}
    if MODELS_DIR.is_dir():
        for d in sorted(MODELS_DIR.iterdir()):
            if not d.is_dir():
                continue
            adapter_dir = d / "lora-adapters"
            if adapter_dir.is_dir():
                models[d.name] = str(adapter_dir)
            elif any(d.glob("*.safetensors")) or any(d.glob("*.bin")) or (d / "adapter_config.json").exists():
                models[d.name] = str(d)
    return models


def _load_model(model_path: str, base_model_id: str = BASE_MODEL_ID):
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = Path(model_path)
    is_peft = path.is_dir() and (path / "adapter_config.json").exists()

    if is_peft:
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id, torch_dtype=dtype, device_map={"": device}, trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map={"": device}, trust_remote_code=True
        )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer


def unload_model(key: str):
    if key in st.session_state:
        del st.session_state[key]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_model(key: str, path_key: str, model_path: str, display_name: str):
    """Load model into session_state[key] if not already loaded for model_path."""
    if key not in st.session_state or st.session_state.get(path_key) != model_path:
        unload_model(key)
        with st.spinner(f"Loading **{display_name}**..."):
            model, tokenizer = _load_model(model_path)
        st.session_state[key] = (model, tokenizer)
        st.session_state[path_key] = model_path
    return st.session_state[key]


def generate_reply(model, tokenizer, turns: list[dict], max_new_tokens: int = 1200) -> str:
    messages = [{"role": "system", "content": THERAPIST_SYSTEM}]
    for t in turns:
        role = "user" if t["role"] == "patient" else "assistant"
        messages.append({"role": role, "content": t["content"]})

    if len(messages) == 1:
        messages.append({"role": "user", "content": "Please begin the session."})

    try:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

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

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return strip_thinking(raw)


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Model Comparison", page_icon="⚖️", layout="wide")
st.title("⚖️ Model Response Comparison")

available_models = discover_models()
model_names = list(available_models.keys())

# ── Sidebar: model selection ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Model A")
    name_a = st.selectbox("Model A", model_names, index=0, key="sel_a")
    path_a = available_models[name_a]
    st.caption(f"`{path_a}`")

    st.divider()

    st.header("Model B")
    default_b = min(1, len(model_names) - 1)
    name_b = st.selectbox("Model B", model_names, index=default_b, key="sel_b")
    path_b = available_models[name_b]
    st.caption(f"`{path_b}`")

    st.divider()
    if st.button("🗑️ Reset Everything"):
        for k in ["json_input", "response_a", "response_b"]:
            st.session_state.pop(k, None)
        st.rerun()

# ── Chat history input ────────────────────────────────────────────────────────
st.subheader("Chat History")
st.caption(
    'Paste a JSON array of turns, e.g. `[{"role": "therapist", "content": "..."}, {"role": "patient", "content": "..."}]`. '
    "The models will generate the **next therapist response** after these turns."
)

EXAMPLE = json.dumps(
    [
        {"role": "therapist", "content": "Hi, how are you feeling today?"},
        {"role": "patient", "content": "Not great, honestly. I've been really anxious."},
    ],
    indent=2,
)

json_input = st.text_area("Paste JSON turns", height=250, key="json_input", placeholder=EXAMPLE)

parse_error = None
turns: list[dict] = []

if json_input.strip():
    try:
        parsed = json.loads(json_input)
        if not isinstance(parsed, list):
            parse_error = "JSON must be an array of turn objects."
        else:
            for i, t in enumerate(parsed):
                if not isinstance(t, dict) or "role" not in t or "content" not in t:
                    parse_error = f"Turn {i} must have `role` and `content` keys."
                    break
                if t["role"] not in ("therapist", "patient"):
                    parse_error = f'Turn {i} role must be "therapist" or "patient", got "{t["role"]}".'
                    break
            else:
                turns = parsed
    except json.JSONDecodeError as e:
        parse_error = f"Invalid JSON: {e}"

if parse_error:
    st.error(parse_error)

# Display parsed turns nicely
if turns:
    st.divider()
    st.markdown("**Parsed conversation:**")
    for t in turns:
        icon = "🧑‍⚕️" if t["role"] == "therapist" else "🗣️"
        label = "Therapist" if t["role"] == "therapist" else "Patient"
        st.markdown(f"{icon} **{label}:** {t['content']}")

st.divider()

# ── Generate ──────────────────────────────────────────────────────────────────
if st.button("🚀 Generate Responses", type="primary", disabled=bool(parse_error) or not json_input.strip()):
    model_a, tok_a = ensure_model("model_a", "path_a", path_a, name_a)
    model_b, tok_b = ensure_model("model_b", "path_b", path_b, name_b)

    with st.spinner("Generating from both models..."):
        st.session_state.response_a = generate_reply(model_a, tok_a, turns)
        st.session_state.response_b = generate_reply(model_b, tok_b, turns)

    st.rerun()

# ── Display responses side-by-side ────────────────────────────────────────────
if "response_a" in st.session_state and "response_b" in st.session_state:
    st.subheader("Responses")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Model A: {name_a}**")
        st.info(st.session_state.response_a)
    with col_b:
        st.markdown(f"**Model B: {name_b}**")
        st.info(st.session_state.response_b)
