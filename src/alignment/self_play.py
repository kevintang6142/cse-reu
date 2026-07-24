"""
Self-Play: Critic-Guided Dialogue Refinement.

Generates simulated CBT dialogues, gets critic feedback, and refines them.
Callable as a module function or standalone script.
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import asyncio
import json
import os
import random
import re
import time
from pathlib import Path

import torch
from openai import AsyncOpenAI, OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer


# ══════════════════════════════════════════════════════════════════════════════
# Configuration defaults
# ══════════════════════════════════════════════════════════════════════════════

BASE_MODEL_NAME = "Qwen/Qwen3.5-4B"

PATIENT_MODEL   = "deepseek-v4-flash"
CRITIC_MODEL    = "deepseek-v4-flash"
MODERATOR_MODEL = "deepseek-v4-flash"
MIN_TURNS       = 5

MAX_TURNS            = 20
THERAPIST_MAX_TOKENS = 256   # real therapist turns avg ~120 tok; tight cap bounds runaway repetition loops
PATIENT_MAX_TOKENS   = 256
THERAPIST_BATCH_SIZE = 4
CRITIC_MAX_TOKENS    = 50000

# Tier-1 decoding guards against repetition. no_repeat_ngram_size hard-blocks the
# model from re-emitting any n-gram already present in the full prompt (prior turns
# included), so it kills verbatim sentence/paragraph loops ACROSS the conversation,
# not just within a single turn. Keep it large (5) so ordinary prose n-grams are not
# banned — small values (e.g. 3) starve normal phrasing and push the model
# off-vocabulary. repetition_penalty softly discourages token-level reuse.
NO_REPEAT_NGRAM_SIZE = 0   # 0 = disabled (testing); if enabling, prefer 5 over 3
REPETITION_PENALTY   = 1.1
# deepseek/OpenAI-style penalty for the API-driven patient agent.
PATIENT_FREQUENCY_PENALTY = 0.5

# Hard cap on therapist prompt length (tokens). Bounds prefill/KV memory no
# matter how long dialogues or refined system prompts grow. Truncated from the
# left so the most recent dialogue is always kept.
THERAPIST_MAX_PROMPT_TOKENS = 16384

# Vignette source: CACTUS distinct-people train split (see src/data/prepare_cactus.py).
# Each self-play iteration randomly samples N_VIGNETTES distinct people from this pool.
CACTUS_TRAIN_PATH = Path("data/processed/cactus_vignettes_train.jsonl")
N_VIGNETTES = 224

N_REFINEMENT_CYCLES = 2
API_CONCURRENCY     = 500

THERAPIST_SYSTEM = (
    "You are a warm, highly skilled CBT (Cognitive Behavioural Therapy) therapist in a one-to-one "
    "session with a client. Conduct the session as a real therapist would, following CBT structure "
    "and staying fully in role at all times.\n\n"
    "SESSION STRUCTURE — progress through these phases over the course of the session; never get "
    "stuck in one phase:\n"
    "1. Open with a brief check-in: a mood check, acknowledge anything that has changed since last "
    "time, and collaboratively set an agenda for what to focus on today.\n"
    "2. Explore the presenting concern with Socratic questioning — draw out the specific thoughts, "
    "emotions, and behaviours and the situations that trigger them.\n"
    "3. Help the client see the links between their thoughts, feelings, and behaviours, and gently "
    "test unhelpful or distorted thinking against the evidence.\n"
    "4. Work collaboratively toward a shared understanding and a concrete, practical coping "
    "strategy or intervention.\n"
    "5. Toward the end, agree a specific between-session homework/action plan, invite the client's "
    "feedback on the session, and close warmly.\n\n"
    "STYLE:\n"
    "- Be warm, empathic, genuine, non-judgmental, and professionally boundaried.\n"
    "- Validate emotions before gently challenging thoughts.\n"
    "- Use plain, everyday language — never clinical jargon or the names of techniques.\n"
    "- Keep every turn short and focused: 2-4 sentences, asking at most one or two questions.\n\n"
    "CRITICAL RULES — follow these on every single turn, without exception:\n"
    "- NEVER repeat yourself. Do not reuse a sentence, question, or phrasing you have already used "
    "earlier in this conversation. Every turn must contain genuinely new content.\n"
    "- Do NOT re-ask anything the client has already answered. Read what they just said, "
    "acknowledge it specifically, and build on it.\n"
    "- Always move the session forward. If a thread is resolved or the client starts repeating "
    "themselves, advance to the next phase rather than circling back to the same point.\n"
    "- Vary your openings and wording; avoid formulaic stock phrases.\n"
    "- Write ONLY the therapist's spoken reply, as natural prose. No stage directions, no "
    "parentheticals, no inner monologue, no lists, no headings, and no meta-commentary.\n"
    "- Respond in English only."
)


# ══════════════════════════════════════════════════════════════════════════════
# CTRS-R Rubric
# ══════════════════════════════════════════════════════════════════════════════

CTSR_ITEMS = [
    {
        "number": 1, "name": "Agenda", "key": "item_1_agenda",
        "criteria": (
            "Did the therapist...\n"
            "• Provide transition to the previous session?\n"
            "• Identify significant events [positive and/or negative] since previous session?\n"
            "• Review Action Plan [complete review may be done as part of the agenda]?\n"
            "• Conduct a mood check?\n"
            "• Identify specific goals or problems to work on during the session?"
        ),
        "anchors": {
            0: "If therapist completed none of the above items",
            1: "If therapist completed one or more but not all of the above items",
            2: "If therapist completed all five of the above items",
            3: "If therapist completed all five of the above items PLUS… Made certain that all items important to the client were addressed and prioritized; Followed the agenda throughout the session unless there was an overt discussion about deviating from the agenda.",
        },
    },
    {
        "number": 2, "name": "Feedback", "key": "item_2_feedback",
        "criteria": (
            "Did the therapist...\n"
            "• Ascertain the client's reaction to the session, the therapist, or the therapeutic process?\n"
            "• Ensure that the client understood and agreed with the treatment plan?\n"
            "• Respond appropriately to feedback?"
        ),
        "anchors": {
            0: "If therapist completed none of the above items",
            1: "If therapist completed one or more but not all of the above items",
            2: "If therapist completed all three of the above items",
            3: "If the therapist completed all three of the above items PLUS… The therapist fluidly requested feedback throughout the session [agenda, transitions, use of techniques, and/or developing an Action Plan].",
        },
    },
    {
        "number": 3, "name": "Understanding", "key": "item_3_understanding",
        "criteria": (
            "Did the therapist...\n"
            "• Demonstrate they generally heard and understood the content of what the client expressed "
            "through repeating, summarizing, etc. what the client said during the session?"
        ),
        "anchors": {
            0: "If therapist did not demonstrate the above item",
            1: "If therapist inconsistently listened and reflected the client's statements",
            2: "If therapist listened and reflected the client's statements throughout the session",
            3: "If the therapist consistently listened and reflected throughout the session PLUS… Therapist demonstrated recognition of understanding the client's emotional state through acknowledgement, reflection, empathy; Discussed the client's emotional state within the context of the conceptualization; Demonstration of emotional state is accomplished by a combination of words, expressions, gestures, tone, and body language throughout the session.",
        },
    },
    {
        "number": 4, "name": "Interpersonal Effectiveness", "key": "item_4_interpersonal_effectiveness",
        "criteria": (
            "Throughout the session, did the therapist…\n"
            "• Demonstrate concern for client and help the client reach their goals?\n"
            "• Provide positive reinforcement for actions taken by the client (e.g. completing action plans)?\n"
            "• Maintain professional and ethical behavior?"
        ),
        "anchors": {
            0: "If therapist completed none of the above items",
            1: "If therapist completed one or two, but not all three of the above items",
            2: "If therapist completed all three of the above items",
            3: "If the therapist completed all three of the above items PLUS… Through words, gestures, and expressions, demonstrated warmth, genuineness, and unconditional acceptance (absence of judgment) by making positive statements about the client's character or characteristics (e.g. strength, determination, caring, vision, values, integrity, etc.)",
        },
    },
    {
        "number": 5, "name": "Collaboration", "key": "item_5_collaboration",
        "criteria": (
            "Did the therapist...\n"
            "• Ask the client for input/agreement when setting the agenda and respond appropriately to the input?\n"
            "• Ask the client for input/agreement when selecting or using CBT techniques and respond appropriately to the input?\n"
            "• Ask the client for input/agreement when determining the Action Plan to be followed between sessions and responded appropriately to the input?"
        ),
        "anchors": {
            0: "If therapist completed none of the above items",
            1: "If therapist completed one or more but not all three of the above items",
            2: "If therapist completed all three of the above items",
            3: "If the therapist completed all three of the above items PLUS… Throughout the session, the therapist made a consistent effort to invite client's participation/agreement on every major decision about the session and responded appropriately. The collaboration resulted in a mutually agreeable direction for the session.",
        },
    },
    {
        "number": 6, "name": "Pacing and Efficient Use of Time", "key": "item_6_pacing",
        "criteria": (
            "Did the therapist...\n"
            "• Allocate appropriate time for transition and agenda setting; intervention(s); feedback and action planning?\n"
            "• Complete the session within 40 – 60 minutes?"
        ),
        "anchors": {
            0: "If therapist completed none of the above items",
            1: "If therapist completed one but not both of the above items",
            2: "If therapist completed both of the above items",
            3: "If the therapist completed both of the above items PLUS… Provided pacing that allowed discussion to seamlessly move through each of the different segments; AND, if needed, made appropriate attempts to limit peripheral or unproductive discussion; AND the session was conducted within 45 – 55 minutes",
        },
    },
    {
        "number": 7, "name": "Guided Discovery", "key": "item_7_guided_discovery",
        "criteria": (
            "Did the therapist...\n"
            "• Design and conduct the session to help the client achieve a cognitive shift regarding agenda items?\n"
            "• Throughout the session, avoid showing bias and avoid use of directions, arguments, or coercion to lead the client "
            "to 'see' things the way the therapist thinks the client should see them?\n"
            "• Assess the cognitive shift following an intervention?"
        ),
        "anchors": {
            0: "If the therapist made no attempt to help the client achieve a cognitive shift",
            1: "If the therapist completed one or more but not all of the above items",
            2: "If the therapist completed all three of the above items",
            3: "If the therapist completed all three of the above items PLUS… Throughout the session, skillfully utilized the process of discovery to help the client arrive at their own conclusions; Assess the potential impact of the cognitive shift on the client's emotions and behaviors.",
        },
    },
    {
        "number": 8, "name": "Focus on Key Cognitions and Behaviors", "key": "item_8_focus_cognitions_behaviors",
        "criteria": (
            "Did the therapist...\n"
            "• Focus on specific cognitions, images, sensations, emotions, behaviors, and or meanings about aspirations "
            "or challenges associated with the sessions Agenda item(s)?"
        ),
        "anchors": {
            0: "If the therapist did not focus on any particular item during the session",
            1: "If the therapist focused on an issue that was unrelated to Agenda items or was unable to elicit specific cognitions, images, sensations, emotions, behaviors, and/or meanings related to Agenda items",
            2: "If the therapist focused on specific cognitions, images, sensations, emotions, behaviors, and/or meanings about aspirations or challenges associated with the sessions Agenda items",
            3: "If the therapist completed the above item PLUS… The items(s) were the most relevant cognitions, images, sensations, emotions, and/or meanings that held greatest promise for a positive impact on the client's aspirations or challenges related to the sessions agenda item(s).",
        },
    },
    {
        "number": 9, "name": "Strategy for Change", "key": "item_9_strategy_for_change",
        "criteria": (
            "Did the therapist...\n"
            "• Discuss evidence-based (CBT) techniques as part of an overall strategy for change with the client?\n"
            "• Select and use at least one identifiable evidence-based technique that was appropriate for the agenda item being addressed?"
        ),
        "anchors": {
            0: "If the therapist did not appear to have any strategy that incorporated use of evidence-based (CBT) techniques",
            1: "If the therapist appeared to have a strategy that did not include use of an appropriate evidence-based (CBT) technique",
            2: "If the therapist discussed an overall strategy for change with the client and used at least one appropriate evidence-based (CBT) technique",
            3: "If the therapist completed both of the items above PLUS… The therapist explained the rationale for use of the technique; Offered other options (if applicable); Obtained the client's agreement to participate in use of the techniques.",
        },
    },
    {
        "number": 10, "name": "Application of CBT Technique", "key": "item_10_application_cbt_technique",
        "criteria": (
            "Did the therapist...\n"
            "• Apply a CBT technique with sufficient skill that the technique was recognizable?\n"
            "• Apply a CBT technique in such a way that it would likely facilitate change in a motivated client?"
        ),
        "anchors": {
            0: "If the therapist attempts to apply a CBT technique was not done with sufficient skill that it was recognizable",
            1: "If the therapist achieved one of the above items but not the other one",
            2: "If the therapist performed the technique with sufficient skill that it accomplished both of the above items",
            3: "If the therapist accomplished both of the above items PLUS The therapist demonstrated good familiarity with the technique; The therapist was comfortable applying the technique; The therapist applied the technique in a technically correct manner (i.e. as the technique is described in the literature).",
        },
    },
    {
        "number": 11, "name": "Action Plan", "key": "item_11_action_plan",
        "criteria": (
            "Did the therapist...\n"
            "• Review the Action Plan from the previous session?\n"
            "• Ask the client to provide input/agreement or incorporate spontaneously offered ideas into the development of a new Action Plan?\n"
            "• Develop an Action Plan based on work done in the current session [and/or continued from a previous session, if applicable] "
            "that, if completed, the Action Plan would answer a question, or help the client to better cope, develop a new skill, or improve their relationships?"
        ),
        "anchors": {
            0: "If the therapist did not complete any of the above items",
            1: "If the therapist completed one or more but not all of the items listed above",
            2: "If the therapist completed all three of the items listed above",
            3: "If the therapist completed all of the items listed above PLUS… The therapist ensured that the client knew what to do, was capable of doing it, and it was specified when, where, how often, and how long to do the Action Plan; and The therapist assessed the reasonable likelihood that the client would complete the Action Plan; and The therapist addressed any challenges or obstacles that would potentially reduce the likelihood of the client completing the Action Plan.",
        },
    },
]

ITEM_KEYS = [it["key"] for it in CTSR_ITEMS]


# ══════════════════════════════════════════════════════════════════════════════
# Vignettes
# ══════════════════════════════════════════════════════════════════════════════

def _patient_system_prompt(intake_form: str, patterns: str) -> str:
    """Build the patient agent's in-character system prompt from a CACTUS persona."""
    return (
        "You are role-playing as a real person attending a CBT (Cognitive Behavioural Therapy) "
        "session as the client. Fully embody the person described below — their background, life "
        "circumstances, and presenting problem are yours.\n\n"
        "=== YOUR PROFILE (intake form) ===\n"
        f"{intake_form}\n\n"
        "=== HOW YOU THINK ===\n"
        f"Your thinking tends to fall into these patterns: {patterns}. "
        "Let them naturally shape how you interpret events and respond, but NEVER name or describe "
        "them — you are not aware of them as 'distortions'; they simply feel true to you.\n\n"
        "=== HOW TO BEHAVE ===\n"
        "- Stay fully in character as this person for the entire conversation.\n"
        "- Speak naturally in everyday language; never use clinical or therapy jargon.\n"
        "- Reveal your problems gradually as the therapist builds rapport — don't dump everything at once.\n"
        "- Be a little guarded at first; let the therapist guide you toward insight rather than handing it over.\n"
        "- React authentically (resistance, relief, doubt, emotion) in a way that fits your profile.\n"
        "- Keep each reply to 2–5 sentences.\n"
        "- Never break character, never mention being an AI, and never reference therapy techniques by name."
    )


def load_vignettes(
    n_vignettes: int = N_VIGNETTES,
    seed: int | None = None,
    train_path: str | Path = CACTUS_TRAIN_PATH,
) -> list[dict]:
    """Randomly sample ``n_vignettes`` distinct CACTUS people for one self-play iteration.

    Each iteration draws a fresh random subset from the train pool (pass a per-iteration
    ``seed`` for reproducibility). Returns vignette dicts with the schema the self-play
    loop expects: ``name`` (unique id), ``background`` (for the critic), ``system_prompt``
    (the patient agent persona).
    """
    train_path = Path(train_path)
    if not train_path.exists():
        raise FileNotFoundError(
            f"CACTUS vignettes not found at {train_path}. "
            "Generate them first with: python -m src.data.prepare_cactus"
        )
    with train_path.open(encoding="utf-8") as f:
        people = [json.loads(line) for line in f if line.strip()]

    rng = random.Random(seed)
    k = min(n_vignettes, len(people))
    sample = rng.sample(people, k)
    if k < n_vignettes:
        print(f"[Self-Play] Requested {n_vignettes} vignettes but pool only has {len(people)}; using {k}.")

    vignettes = []
    for p in sample:
        patterns = ", ".join(p["patterns"])
        background = f"{p['intake_form']}\n\nCognitive distortion patterns: {patterns}"
        vignettes.append({
            "name": p["id"],
            "background": background,
            "system_prompt": _patient_system_prompt(p["intake_form"], patterns),
        })
    print(f"[Self-Play] Sampled {len(vignettes)} CACTUS vignettes from {len(people)} train people (seed={seed})")
    return vignettes


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_therapist_model(model_path: str, base_model_name: str | None = None):
    """Load a therapist model for inference (full or PEFT adapter)."""
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    path = Path(model_path)

    is_peft = (path / "adapter_config.json").exists()
    if is_peft:
        from peft import PeftModel
        assert base_model_name, "base_model_name required when loading a PEFT adapter"
        print("[Self-Play] Loading base + PEFT adapter and merging…")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name, dtype=dtype, device_map="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
    else:
        print(f"[Self-Play] Loading model from {model_path}…")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, device_map="auto", trust_remote_code=True
        )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[Self-Play] Loaded {total_params:.1f}B parameters.")
    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# Conversation engine
# ══════════════════════════════════════════════════════════════════════════════

def strip_thinking(text: str) -> str:
    """Strip Qwen3 <think>...</think> blocks and leaked inline scratchpad."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = strip_meta(cleaned)
    return cleaned.strip()


def strip_meta(text: str) -> str:
    """Remove leaked italic-parenthetical scratchpad, e.g. ``*(Self-Correction: ...)*``
    or ``*(Wait, I am still repeating...)*``. These are meta-commentary the model emits
    inline (not inside <think> tags), and they both pollute SFT data and confuse the
    patient/moderator agents. Collapse any whitespace the removal leaves behind."""
    cleaned = re.sub(r"\*\(.*?\)\*", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def to_hf_messages(turns: list[dict], system_prompt: str) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    for t in turns:
        role = "user" if t["role"] == "patient" else "assistant"
        messages.append({"role": role, "content": t["content"]})
    return messages


def to_patient_messages(turns: list[dict], system_prompt: str) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    for t in turns:
        role = "user" if t["role"] == "therapist" else "assistant"
        messages.append({"role": role, "content": t["content"]})
    return messages


def _build_therapist_prompt(tokenizer, turns: list[dict], system_prompt: str) -> str:
    """Build one chat-template prompt for therapist inference."""
    messages = to_hf_messages(turns, system_prompt)
    if len(messages) == 1:
        messages.append({"role": "user", "content": "Please begin the session."})

    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def _therapist_stop_ids(tokenizer) -> list[int]:
    """Return valid EOS/turn-end token ids for generation."""
    ids = [tokenizer.eos_token_id]
    try:
        ids.append(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    except Exception:
        pass
    return sorted({int(x) for x in ids if isinstance(x, int) and x >= 0})


def therapist_turn_batch(
    model,
    tokenizer,
    turns_batch: list[list[dict]],
    system_prompts: list[str],
    batch_size: int = THERAPIST_BATCH_SIZE,
    temperature: float = 0.7,
) -> list[str]:
    """Generate multiple therapist utterances in batched GPU calls.

    This is the throughput-critical path. Running many independent model.generate(...)
    calls serially underutilizes the GPU; batching lets one forward pass serve multiple
    active conversations. Keep batch_size modest because generation speed is bounded by
    the longest sequence in each batch and by available VRAM.
    """
    if len(turns_batch) != len(system_prompts):
        raise ValueError("turns_batch and system_prompts must have the same length")
    if not turns_batch:
        return []
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    prompt_texts = [
        _build_therapist_prompt(tokenizer, turns, system_prompt)
        for turns, system_prompt in zip(turns_batch, system_prompts)
    ]

    # Bucket by prompt length to reduce padding waste, then restore original order.
    order = sorted(range(len(prompt_texts)), key=lambda i: len(prompt_texts[i]))
    responses: list[str | None] = [None] * len(prompt_texts)
    stop_ids = _therapist_stop_ids(tokenizer)

    old_padding_side = getattr(tokenizer, "padding_side", "right")
    old_truncation_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.padding_side = "left"  # decoder-only models generate more correctly with left padding
    tokenizer.truncation_side = "left"  # keep the most recent context if a prompt is over-length

    n_batches = (len(order) + batch_size - 1) // batch_size
    try:
        for start in range(0, len(order), batch_size):
            batch_indices = order[start:start + batch_size]
            batch_prompts = [prompt_texts[i] for i in batch_indices]
            done = min(start + batch_size, len(order))
            print(
                f"\r      therapist batch {start // batch_size + 1}/{n_batches} "
                f"({done}/{len(order)} convos)",
                end="", flush=True,
            )

            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=THERAPIST_MAX_PROMPT_TOKENS,
            ).to(model.device)

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=THERAPIST_MAX_TOKENS,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=REPETITION_PENALTY,
                    no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                    eos_token_id=stop_ids,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )

            # With left padding, generated tokens begin after the padded prompt width.
            gen_start = inputs["input_ids"].shape[1]
            new_token_ids = output_ids[:, gen_start:]
            decoded = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)

            for original_idx, raw in zip(batch_indices, decoded):
                responses[original_idx] = strip_thinking(raw)

            del inputs, output_ids, new_token_ids
            torch.cuda.empty_cache()

        print(f"\r      therapist batches done ({n_batches}/{n_batches})        ", flush=True)

    finally:
        tokenizer.padding_side = old_padding_side
        tokenizer.truncation_side = old_truncation_side

    return [r if r is not None else "" for r in responses]


def therapist_turn(model, tokenizer, turns: list[dict], system_prompt: str) -> str:
    """Backward-compatible single-conversation wrapper."""
    return therapist_turn_batch(model, tokenizer, [turns], [system_prompt], batch_size=1)[0]


def _prompt_for_idx(system_prompt: str | list[str], idx: int) -> str:
    return system_prompt[idx] if isinstance(system_prompt, list) else system_prompt


def append_batched_therapist_turns(
    model,
    tokenizer,
    all_turns: list[list[dict]],
    active: list[int],
    system_prompt: str | list[str],
    therapist_batch_size: int = THERAPIST_BATCH_SIZE,
    temperature: float = 0.7,
) -> None:
    """Append one therapist turn to every active conversation using batched inference."""
    if not active:
        return
    turns_batch = [all_turns[idx] for idx in active]
    prompt_batch = [_prompt_for_idx(system_prompt, idx) for idx in active]
    responses = therapist_turn_batch(
        model,
        tokenizer,
        turns_batch,
        prompt_batch,
        batch_size=therapist_batch_size,
        temperature=temperature,
    )
    for idx, t_msg in zip(active, responses):
        all_turns[idx].append({"role": "therapist", "content": t_msg})


async def patient_turn_async(
    client: AsyncOpenAI, sem: asyncio.Semaphore, vignette: dict, turns: list[dict],
) -> str:
    messages = to_patient_messages(turns, vignette["system_prompt"])
    async with sem:
        resp = await client.chat.completions.create(
            model=PATIENT_MODEL,
            messages=messages,
            max_tokens=PATIENT_MAX_TOKENS,
            temperature=0.85,
            frequency_penalty=PATIENT_FREQUENCY_PENALTY,
            extra_body={"thinking": {"type": "disabled"}},
        )
    return strip_meta((resp.choices[0].message.content or "").strip())


async def should_end_session_async(
    client: AsyncOpenAI, sem: asyncio.Semaphore, turns: list[dict],
) -> bool:
    transcript_text = "\n".join(
        f"{'Therapist' if t['role'] == 'therapist' else 'Patient'}: {t['content']}"
        for t in turns
    )
    prompt = (
        "Transcript of a CBT therapy session:\n\n"
        f"{transcript_text}\n\n"
        "End the session (answer 'Yes') if EITHER condition holds; otherwise 'No'.\n\n"
        "A) Fully complete: the therapist has deeply explored the patient's concerns, applied a "
        "specific CBT technique, agreed a concrete homework/action plan (not vague), begun wrapping "
        "up, AND the patient has no concerns left. If still exploring, no technique/plan yet, or not "
        "wrapping up, A is not met.\n\n"
        "B) Genuine farewell: either the therapist OR the patient is actually saying goodbye to the "
        "other to end today's session (e.g. 'Goodbye, take care, see you next week'). Do NOT count a goodbye "
        "that appears inside a story, quote, or memory the patient is recounting (e.g. 'she said "
        "goodbye and walked out') — that is narration, not closing. When unsure, treat it as an "
        "anecdote and do not end.\n\n"
        "When in doubt, answer 'No'. Answer with ONLY 'Yes' or 'No'."
    )
    async with sem:
        resp = await client.chat.completions.create(
            model=MODERATOR_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3,
            temperature=0.0,
            extra_body={"thinking": {"type": "disabled"}},
        )
    answer = (resp.choices[0].message.content or "").strip().lower()
    return answer.startswith("yes")


async def run_all_interviews(
    model, tokenizer, client: AsyncOpenAI, sem: asyncio.Semaphore,
    vignettes: list[dict], system_prompt: str | list[str],
    therapist_batch_size: int = THERAPIST_BATCH_SIZE,
) -> list[dict]:
    """Run all vignettes in lockstep."""
    n = len(vignettes)
    all_turns: list[list[dict]] = [[] for _ in range(n)]
    active = list(range(n))

    print(
        f"  Round 0: therapist opening for {len(active)} vignettes "
        f"(batch_size={therapist_batch_size})...",
        flush=True,
    )
    append_batched_therapist_turns(
        model, tokenizer, all_turns, active, system_prompt, therapist_batch_size
    )
    print(f"    done ({len(active)} openings)              ")

    for round_num in range(1, MAX_TURNS):
        if not active:
            break

        print(f"  Round {round_num}: patient×{len(active)}...", end=" ", flush=True)
        patient_tasks = [
            patient_turn_async(client, sem, vignettes[idx], all_turns[idx])
            for idx in active
        ]
        patient_responses = await asyncio.gather(*patient_tasks)
        for i, idx in enumerate(active):
            all_turns[idx].append({"role": "patient", "content": patient_responses[i]})

        print(f"therapist×{len(active)} batch_size={therapist_batch_size}...", flush=True)
        append_batched_therapist_turns(
            model, tokenizer, all_turns, active, system_prompt, therapist_batch_size
        )
        print("    therapist done              ", flush=True)

        eligible = [
            idx for idx in active
            if (len(all_turns[idx]) + 1) // 2 >= MIN_TURNS
        ]
        ended = set()
        if eligible:
            print(f"moderator×{len(eligible)}...", end=" ", flush=True)
            mod_tasks = [
                should_end_session_async(client, sem, all_turns[idx])
                for idx in eligible
            ]
            mod_results = await asyncio.gather(*mod_tasks)
            for i, idx in enumerate(eligible):
                if mod_results[i]:
                    ended.add(idx)

        active = [idx for idx in active if idx not in ended]
        print(f"ended={len(ended)}, active={len(active)}")

    results = []
    for idx in range(n):
        results.append({
            "vignette": vignettes[idx]["name"],
            "turns": all_turns[idx],
        })
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Critic
# ══════════════════════════════════════════════════════════════════════════════

def build_critic_prompt(vignette: dict, turns: list[dict]) -> str:
    transcript_text = "\n".join(
        f"{'Therapist' if t['role'] == 'therapist' else 'Patient'}: {t['content']}"
        for t in turns
    )
    rubric_text = ""
    for item in CTSR_ITEMS:
        anchors_text = "\n".join(f"      {k} = {v}" for k, v in item["anchors"].items())
        rubric_text += (
            f"  Item {item['number']}: {item['name']} (key: {item['key']})\n"
            f"    Criteria:\n{item['criteria']}\n"
            f"    Score anchors:\n{anchors_text}\n\n"
        )

    return (
        "You are a CBT clinical supervisor providing detailed feedback to a therapist-in-training.\n"
        "Evaluate the therapist's performance using the Cognitive Therapy Rating Scale – Revised (CTRS-R).\n\n"
        f"Patient background:\n{vignette['background']}\n\n"
        f"Session transcript:\n{transcript_text}\n\n"
        f"CTRS-R Rubric:\n{rubric_text}\n"
        "Score the therapist on ALL 11 items. For each item, provide reasoning "
        "with specific transcript references, a score, and actionable improvement suggestions.\n\n"
        "Output ONLY a JSON object with this exact structure:\n"
        "{\n"
        '  "item_1_agenda": {"reasoning": "...", "score": N, "improvement": "..."},\n'
        '  "item_2_feedback": {"reasoning": "...", "score": N, "improvement": "..."},\n'
        '  "item_3_understanding": {"reasoning": "...", "score": N, "improvement": "..."},\n'
        '  "item_4_interpersonal_effectiveness": {"reasoning": "...", "score": N, "improvement": "..."},\n'
        '  "item_5_collaboration": {"reasoning": "...", "score": N, "improvement": "..."},\n'
        '  "item_6_pacing": {"reasoning": "...", "score": N, "improvement": "..."},\n'
        '  "item_7_guided_discovery": {"reasoning": "...", "score": N, "improvement": "..."},\n'
        '  "item_8_focus_cognitions_behaviors": {"reasoning": "...", "score": N, "improvement": "..."},\n'
        '  "item_9_strategy_for_change": {"reasoning": "...", "score": N, "improvement": "..."},\n'
        '  "item_10_application_cbt_technique": {"reasoning": "...", "score": N, "improvement": "..."},\n'
        '  "item_11_action_plan": {"reasoning": "...", "score": N, "improvement": "..."}\n'
        "}\n\n"
        "Rules:\n"
        "- Output ONLY the JSON object, no other text.\n"
        "- Each score must be an integer 0–3.\n"
        "- Each reasoning must be concise (2–4 sentences) with specific transcript references.\n"
        "- Each improvement must be a specific, actionable suggestion for what to do differently."
    )


def parse_critic_json(raw: str) -> dict | None:
    """Try to parse critic response as JSON. Returns dict or None."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None


def format_critic_feedback(raw_json: str) -> str:
    """Format structured critic JSON into readable feedback for the system prompt."""
    data = parse_critic_json(raw_json)
    if data is None:
        return raw_json

    parts = []
    for item in CTSR_ITEMS:
        key = item["key"]
        if key in data and isinstance(data[key], dict):
            entry = data[key]
            parts.append(
                f"Item {item['number']} ({item['name']}): {entry.get('score', '?')}/3\n"
                f"  Reasoning: {entry.get('reasoning', 'N/A')}\n"
                f"  Improvement: {entry.get('improvement', 'N/A')}"
            )
    return "\n".join(parts)


async def get_critic_feedback_raw(
    client: AsyncOpenAI, sem: asyncio.Semaphore, vignette: dict, turns: list[dict],
) -> str:
    """Get raw critic response for a single dialogue."""
    prompt = build_critic_prompt(vignette, turns)
    async with sem:
        resp = await client.chat.completions.create(
            model=CRITIC_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=CRITIC_MAX_TOKENS,
            temperature=0.3,
            extra_body={"thinking": {"type": "disabled"}},
        )
    return (resp.choices[0].message.content or "").strip()


async def get_all_critic_feedback(
    client: AsyncOpenAI, sem: asyncio.Semaphore,
    transcripts: list[dict], vignettes: list[dict],
) -> tuple[list[str], list[str]]:
    """
    Get critic feedback for all transcripts.
    Returns (formatted_feedbacks, raw_responses).
    """
    vignette_map = {v["name"]: v for v in vignettes}
    tasks = [
        get_critic_feedback_raw(client, sem, vignette_map[t["vignette"]], t["turns"])
        for t in transcripts
    ]
    print(f"  Firing {len(tasks)} critic calls (semaphore={API_CONCURRENCY})...", flush=True)
    raw_results = await asyncio.gather(*tasks)
    print(f"  All {len(tasks)} critic calls complete.")

    formatted = [format_critic_feedback(r) for r in raw_results]
    return formatted, raw_results


# ══════════════════════════════════════════════════════════════════════════════
# Refined system prompt
# ══════════════════════════════════════════════════════════════════════════════

def build_refined_system_prompt(
    base_system: str,
    previous_attempts: list[dict],
) -> str:
    parts = [base_system]
    # Only condition on the MOST RECENT attempt + its critique. Embedding every
    # prior attempt makes the system prompt — and therefore peak generation VRAM —
    # grow with each refinement cycle; the latest attempt carries the actionable
    # signal the therapist needs to improve.
    if previous_attempts:
        attempt = previous_attempts[-1]
        transcript_text = "\n".join(
            f"<{'your_turn' if t['role'] == 'therapist' else 'client_turn'}>{t['content']}</{'your_turn' if t['role'] == 'therapist' else 'client_turn'}>"
            for t in attempt["turns"]
        )
        parts.append(f"\n<previous_attempt>\n{transcript_text}\n</previous_attempt>")
        parts.append(f"\n<critic_feedback>\n{attempt['feedback']}\n</critic_feedback>")
    parts.append(
        "\nUse the lessons from the previous attempts and feedback."
        "\nStart a new conversation with the patient from the beginning."
        "\nRespond only as the therapist. Do NOT prefix your response with any role label."
    )
    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# Self-play iteration
# ══════════════════════════════════════════════════════════════════════════════

async def self_play_iteration(
    model, tokenizer, client: AsyncOpenAI, sem: asyncio.Semaphore,
    vignettes: list[dict], n_refinement_cycles: int,
    therapist_batch_size: int = THERAPIST_BATCH_SIZE,
) -> tuple[list[dict], list[str], list[dict]]:
    """
    Run one full self-play iteration.

    Returns (final_transcripts, final_critic_raw_responses, all_rounds) where
    all_rounds is a list of per-round snapshots, each
    {"round": int, "transcripts": [...], "critiques": [raw_str, ...]} with each
    transcript ALIGNED to the critique that scored it. Round 0 is the initial
    generation; the final round is the version returned. SFT uses only the final
    version; KTO uses every round.
    """
    n = len(vignettes)
    history: list[list[dict]] = [[] for _ in range(n)]
    all_rounds: list[dict] = []

    # ── Initial generation ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"INITIAL GENERATION ({n} vignettes)")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    transcripts = await run_all_interviews(
        model, tokenizer, client, sem, vignettes, THERAPIST_SYSTEM, therapist_batch_size
    )
    elapsed = time.perf_counter() - t0
    print(f"  Initial generation done in {elapsed:.1f}s")

    last_raw_critiques: list[str] = []

    # ── Refinement cycles ─────────────────────────────────────────────────────
    for cycle in range(n_refinement_cycles):
        print(f"\n{'='*60}")
        print(f"REFINEMENT CYCLE {cycle + 1}/{n_refinement_cycles}")
        print(f"{'='*60}")

        print("\n  Getting critic feedback...")
        t0 = time.perf_counter()
        feedbacks, raw_critiques = await get_all_critic_feedback(
            client, sem, transcripts, vignettes
        )
        last_raw_critiques = raw_critiques
        # `transcripts` is the exact version that was just critiqued → save the
        # aligned (transcript, critique) snapshot for this round.
        all_rounds.append({"round": cycle, "transcripts": transcripts, "critiques": raw_critiques})
        elapsed = time.perf_counter() - t0
        print(f"  Critic feedback done in {elapsed:.1f}s")

        for idx in range(n):
            history[idx].append({
                "turns": transcripts[idx]["turns"],
                "feedback": feedbacks[idx],
            })

        refined_prompts = [
            build_refined_system_prompt(THERAPIST_SYSTEM, history[idx])
            for idx in range(n)
        ]

        print("\n  Regenerating dialogues with feedback...")
        t0 = time.perf_counter()

        transcripts = await run_all_interviews(
            model,
            tokenizer,
            client,
            sem,
            vignettes,
            refined_prompts,
            therapist_batch_size,
        )

        elapsed = time.perf_counter() - t0
        print(f"  Regeneration done in {elapsed:.1f}s")

    # Critique the FINAL version too. The last regeneration was never scored, so
    # previously the saved critiques described the second-to-last version. This
    # closes that gap and gives the final round its own aligned critique.
    print("\n  Getting critic feedback on FINAL version...")
    _, final_raw_critiques = await get_all_critic_feedback(
        client, sem, transcripts, vignettes
    )
    all_rounds.append({
        "round": n_refinement_cycles,
        "transcripts": transcripts,
        "critiques": final_raw_critiques,
    })

    return transcripts, final_raw_critiques, all_rounds


# ══════════════════════════════════════════════════════════════════════════════
# Chunking for long conversations
# ══════════════════════════════════════════════════════════════════════════════

MAX_SEQ_LENGTH = 4096
SUMMARY_MODEL  = "deepseek-v4-flash"

USER_START_MSG = {"role": "user", "content": "Start the session."}


def ensure_user_first(messages: list[dict]) -> list[dict]:
    """If the first non-system turn is assistant, prepend a user message."""
    if len(messages) > 1 and messages[0]["role"] == "system" and messages[1]["role"] == "assistant":
        return [messages[0], USER_START_MSG] + messages[1:]
    return messages


def count_tokens_for_chunking(messages: list[dict], tok) -> int:
    """Count tokens for a message list using the model's chat template."""
    messages = ensure_user_first(messages)
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return len(tok.encode(text))


def summarize_previous_turns(turns: list[dict], ds_client: OpenAI) -> str:
    """Ask DeepSeek for a concise summary of prior conversation turns."""
    conversation_text = "\n".join(
        f"{'Therapist' if m['role'] == 'assistant' else 'Patient'}: {m['content']}"
        for m in turns
    )
    resp = ds_client.chat.completions.create(
        model=SUMMARY_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Summarize the following therapy conversation concisely in at most "
                    "1000 words. Focus on key topics discussed, patient concerns, "
                    "emotional state, and therapeutic progress. Output only the summary."
                ),
            },
            {"role": "user", "content": conversation_text},
        ],
        max_tokens=1500,
        temperature=0.3,
    )
    return (resp.choices[0].message.content or "").strip()


def chunk_conversation(messages: list[dict], tok, ds_client: OpenAI) -> list[dict]:
    """Split one conversation into chunks each ≤ MAX_SEQ_LENGTH tokens.

    Chunk 1 : system prompt + complete turns until hitting the limit.
    Chunk 2+: system prompt (with summary of ALL prior turns appended)
              + complete turns until hitting the limit.
    """
    system_msg = messages[0]
    turns = messages[1:]

    chunks = []
    turn_idx = 0
    previous_turns = []

    while turn_idx < len(turns):
        if not chunks:
            sys_msg = system_msg
        else:
            summary = summarize_previous_turns(previous_turns, ds_client)
            sys_msg = {
                "role": "system",
                "content": (
                    system_msg["content"]
                    + "\n\n[Summary of conversation so far]\n"
                    + summary
                ),
            }

        chunk_msgs = [sys_msg]

        while turn_idx < len(turns):
            candidate = chunk_msgs + [turns[turn_idx]]
            if count_tokens_for_chunking(candidate, tok) > MAX_SEQ_LENGTH and len(chunk_msgs) > 1:
                break
            chunk_msgs.append(turns[turn_idx])
            turn_idx += 1

        previous_turns.extend(chunk_msgs[1:])
        chunks.append({"messages": ensure_user_first(chunk_msgs)})

    return chunks


def chunk_sft_records(sft_records: list[dict], tok, ds_client: OpenAI) -> list[dict]:
    """Chunk any records that exceed MAX_SEQ_LENGTH tokens."""
    out = []
    needs_chunking = 0
    for i, rec in enumerate(sft_records):
        msgs = ensure_user_first(rec["messages"])
        if count_tokens_for_chunking(msgs, tok) <= MAX_SEQ_LENGTH:
            out.append({"messages": msgs})
        else:
            needs_chunking += 1
            out.extend(chunk_conversation(msgs, tok, ds_client))
    print(f"[Self-Play] Chunking: {len(sft_records)} conversations → {len(out)} records "
          f"({needs_chunking} needed chunking)")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Conversion to SFT format
# ══════════════════════════════════════════════════════════════════════════════

def dialogue_to_sft_format(dialogue: dict) -> dict:
    messages = [{"role": "system", "content": THERAPIST_SYSTEM}]
    for turn in dialogue["turns"]:
        if turn["role"] == "therapist":
            messages.append({"role": "assistant", "content": turn["content"]})
        else:
            messages.append({"role": "user", "content": turn["content"]})
    return {"messages": messages}


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_self_play(
    adapter_path: str | None,
    output_jsonl: str | Path,
    output_transcripts: str | Path | None = None,
    output_critiques: str | Path | None = None,
    output_rounds_transcripts: str | Path | None = None,
    output_rounds_critiques: str | Path | None = None,
    base_model_name: str = BASE_MODEL_NAME,
    n_vignettes: int = N_VIGNETTES,
    vignette_seed: int | None = None,
    n_refinement_cycles: int = N_REFINEMENT_CYCLES,
    api_concurrency: int = API_CONCURRENCY,
    therapist_batch_size: int = THERAPIST_BATCH_SIZE,
) -> Path:
    """
    Run self-play and save results.

    Args:
        adapter_path: Path to LoRA adapter dir, or None to use base model.
        output_jsonl: Where to save SFT-format training data (final version only).
        output_transcripts: Where to save the FINAL raw transcripts (JSONL) — SFT uses these.
        output_critiques: Where to save the FINAL critic judgments (JSONL), aligned to
            output_transcripts.
        output_rounds_transcripts / output_rounds_critiques: Where to save EVERY round's
            transcript and its aligned critique (initial + each refinement). ~n_vignettes ×
            (n_refinement_cycles + 1) rows each, index-aligned. KTO consumes these.
        base_model_name: HuggingFace model ID for base model.
        therapist_batch_size: Number of active conversations to generate in each local
            HuggingFace batch. Increase until VRAM becomes the bottleneck.

    Returns the output_jsonl path.
    """
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    if adapter_path is not None:
        print(f"[Self-Play] Loading adapter: {adapter_path}")
        model, tokenizer = load_therapist_model(adapter_path, base_model_name=base_model_name)
    else:
        print(f"[Self-Play] Loading base model: {base_model_name}")
        model, tokenizer = load_therapist_model(base_model_name)

    # Load vignettes — fresh random sample of CACTUS people for this iteration
    vignettes = load_vignettes(n_vignettes, seed=vignette_seed)

    # Setup async client
    client = AsyncOpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    sem = asyncio.Semaphore(api_concurrency)

    # Run
    print(f"\n[Self-Play] Starting self-play ({n_refinement_cycles} refinement cycles)...")
    t0 = time.perf_counter()
    transcripts, raw_critiques, all_rounds = asyncio.run(
        self_play_iteration(
            model, tokenizer, client, sem, vignettes, n_refinement_cycles, therapist_batch_size
        )
    )
    elapsed = time.perf_counter() - t0
    print(f"[Self-Play] Done in {elapsed:.1f}s — {len(transcripts)} dialogues")

    # Save SFT-format data (with chunking for long conversations)
    sft_records = [dialogue_to_sft_format(d) for d in transcripts]

    # Reuse the already-loaded tokenizer for token counting; it shares the
    # base model's vocab/chat template. Sync DeepSeek client handles summaries.
    ds_sync = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    sft_records = chunk_sft_records(sft_records, tokenizer, ds_sync)

    with open(output_jsonl, "w") as f:
        for record in sft_records:
            f.write(json.dumps(record) + "\n")
    print(f"[Self-Play] SFT data saved → {output_jsonl} ({len(sft_records)} records)")

    # Save raw transcripts
    if output_transcripts:
        output_transcripts = Path(output_transcripts)
        output_transcripts.parent.mkdir(parents=True, exist_ok=True)
        with open(output_transcripts, "w") as f:
            for t in transcripts:
                f.write(json.dumps(t) + "\n")
        print(f"[Self-Play] Transcripts saved → {output_transcripts}")

    # Save critic judgments
    if output_critiques and raw_critiques:
        output_critiques = Path(output_critiques)
        output_critiques.parent.mkdir(parents=True, exist_ok=True)
        with open(output_critiques, "w") as f:
            for i, raw in enumerate(raw_critiques):
                record = {
                    "vignette": transcripts[i]["vignette"],
                    "raw_critique": raw,
                    "parsed": parse_critic_json(raw),
                }
                f.write(json.dumps(record) + "\n")
        print(f"[Self-Play] Critiques saved → {output_critiques}")

    # Save EVERY round (initial + refinements + final), transcript aligned to its
    # own critique. Two index-aligned files so build_kto_records can consume them
    # directly. SFT ignores these; KTO uses them for the full quality gradient.
    if output_rounds_transcripts and output_rounds_critiques:
        ort = Path(output_rounds_transcripts)
        orc = Path(output_rounds_critiques)
        ort.parent.mkdir(parents=True, exist_ok=True)
        orc.parent.mkdir(parents=True, exist_ok=True)
        n_rows = 0
        with open(ort, "w") as ft, open(orc, "w") as fc:
            for rd in all_rounds:
                r = rd["round"]
                for t, raw in zip(rd["transcripts"], rd["critiques"]):
                    ft.write(json.dumps(
                        {"vignette": t["vignette"], "round": r, "turns": t["turns"]}
                    ) + "\n")
                    fc.write(json.dumps(
                        {"vignette": t["vignette"], "round": r,
                         "raw_critique": raw, "parsed": parse_critic_json(raw)}
                    ) + "\n")
                    n_rows += 1
        print(f"[Self-Play] All rounds saved → {ort} + {orc} "
              f"({n_rows} dialogues across {len(all_rounds)} rounds)")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return output_jsonl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run self-play dialogue generation")
    parser.add_argument("--adapter-path", default=None, help="Path to LoRA adapter (None=base model)")
    parser.add_argument("--output-jsonl", required=True, help="Output SFT JSONL path")
    parser.add_argument("--output-transcripts", default=None, help="Output transcripts JSONL path")
    parser.add_argument("--output-critiques", default=None, help="Output critic judgments JSONL path")
    parser.add_argument("--output-rounds-transcripts", default=None,
                        help="Output ALL-rounds transcripts JSONL (initial+refinements+final)")
    parser.add_argument("--output-rounds-critiques", default=None,
                        help="Output ALL-rounds critiques JSONL, aligned to rounds transcripts")
    parser.add_argument("--base-model", default=BASE_MODEL_NAME)
    parser.add_argument("--n-vignettes", type=int, default=N_VIGNETTES,
                        help="Distinct CACTUS people to sample for this iteration")
    parser.add_argument("--vignette-seed", type=int, default=None,
                        help="Seed for the per-iteration vignette sample (None=nondeterministic)")
    parser.add_argument("--n-refinement-cycles", type=int, default=N_REFINEMENT_CYCLES)
    parser.add_argument("--api-concurrency", type=int, default=API_CONCURRENCY)
    parser.add_argument(
        "--therapist-batch-size",
        type=int,
        default=THERAPIST_BATCH_SIZE,
        help="Batch size for local HuggingFace therapist model.generate calls",
    )
    args = parser.parse_args()

    run_self_play(
        adapter_path=args.adapter_path,
        output_jsonl=args.output_jsonl,
        output_transcripts=args.output_transcripts,
        output_critiques=args.output_critiques,
        output_rounds_transcripts=args.output_rounds_transcripts,
        output_rounds_critiques=args.output_rounds_critiques,
        base_model_name=args.base_model,
        n_vignettes=args.n_vignettes,
        vignette_seed=args.vignette_seed,
        n_refinement_cycles=args.n_refinement_cycles,
        api_concurrency=args.api_concurrency,
        therapist_batch_size=args.therapist_batch_size,
    )
