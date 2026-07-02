"""
chatbot_ui.py — Streamlit chatbot for talking to base or fine-tuned therapist models.

Usage:
    uv run streamlit run src/scripts/chatbot_ui.py
"""

import gc
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


# ── Model helpers ─────────────────────────────────────────────────────────────

def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def discover_models() -> dict[str, str]:
    """Return {display_name: path_or_id} for all available models."""
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


def unload_model():
    """Free GPU memory from the previously loaded model."""
    if "loaded_model" in st.session_state:
        del st.session_state["loaded_model"]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(model_path: str, base_model_id: str = BASE_MODEL_ID):
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


def generate_reply(model, tokenizer, turns: list[dict], max_new_tokens: int = 1200) -> str:
    messages = [{"role": "system", "content": THERAPIST_SYSTEM}]
    for t in turns:
        role = "user" if t["role"] == "user" else "assistant"
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

st.set_page_config(page_title="Therapist Chatbot", page_icon="🧠", layout="centered")
st.title("🧠 Therapist Chatbot")

available_models = discover_models()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Selection")
    selected_name = st.selectbox(
        "Choose a model",
        options=list(available_models.keys()),
        index=0,
    )
    selected_path = available_models[selected_name]
    st.caption(f"`{selected_path}`")

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.pop("messages", None)
        st.rerun()

# ── Detect model switch ──────────────────────────────────────────────────────
if st.session_state.get("active_model_path") != selected_path:
    unload_model()
    st.session_state["active_model_path"] = selected_path
    st.session_state.pop("messages", None)

# ── Ensure model is loaded ────────────────────────────────────────────────────
if "loaded_model" not in st.session_state or st.session_state.get("active_model_path") != st.session_state.get("loaded_model_path"):
    with st.spinner(f"Loading **{selected_name}**..."):
        model, tokenizer = load_model(selected_path)
    st.session_state["loaded_model"] = (model, tokenizer)
    st.session_state["loaded_model_path"] = selected_path
else:
    model, tokenizer = st.session_state["loaded_model"]

st.sidebar.success(f"**{selected_name}** loaded")

# ── Chat state ────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

    with st.spinner("Therapist is thinking..."):
        opening = generate_reply(model, tokenizer, [])
    st.session_state.messages.append({"role": "assistant", "content": opening})

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── User input ────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Your response..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Therapist is thinking..."):
            reply = generate_reply(model, tokenizer, st.session_state.messages)
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
