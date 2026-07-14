"""
Build a "condensed" SFT dataset from full-length conversations.

From each conversation, sample non-overlapping windows of 3 patient->therapist
exchange pairs (6 messages: u/a/u/a/u/a). For windows that don't start at the
beginning of the conversation, a DeepSeek summary of all preceding turns is
appended to the system prompt (same method as chunk_conversation in
self_play.py / 02_preprocessing.ipynb).

Therapist backchannels ("Mm-hmm.", "Yeah.", "Okay.") are stripped before
sampling so SFT never trains on them as complete responses, and each window's
final therapist turn must be >= MIN_FINAL_WORDS words.

Usage:
    python -m src.alignment.condense --input data/processed/combined_train.jsonl \
                                     --output data/processed/condensed_train.jsonl \
                                     --seed 42
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import os
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


# ── Defaults ──────────────────────────────────────────────────────────────────
SUMMARY_MODEL     = "deepseek-v4-flash"
EXCHANGES_PER_WIN = 3     # patient->therapist pairs per window
MSGS_PER_WINDOW   = 20    # ~1 window per this many non-system messages
MIN_FINAL_WORDS   = 5     # window's final therapist turn must have >= this many words
SUMMARY_WORKERS   = 8

# Assistant messages whose every token is in this set (and <= 4 words) are
# treated as pure backchannels and dropped.
BACKCHANNEL_VOCAB = {
    "mm", "mmm", "hmm", "hm", "mhm", "mmhmm", "uh", "huh",
    "yeah", "yes", "yep", "no", "nope",
    "okay", "ok", "right", "alright", "all", "sure", "oh",
    "thanks", "thank", "you", "cool", "great", "good", "gotcha", "wow",
}
MAX_BACKCHANNEL_WORDS = 4


def load_jsonl(path: str | Path) -> list[dict]:
    """Load JSONL records."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── Backchannel cleaning ──────────────────────────────────────────────────────

def is_backchannel(content: str) -> bool:
    """True if an assistant message is a pure acknowledgement like 'Mm-hmm.'"""
    tokens = re.sub(r"[^a-z0-9]+", " ", content.lower()).split()
    if not tokens or len(tokens) > MAX_BACKCHANNEL_WORDS:
        return False
    return all(t in BACKCHANNEL_VOCAB for t in tokens)


def clean_conversation(messages: list[dict]) -> tuple[dict, list[dict]]:
    """Drop assistant backchannels, then merge adjacent same-role messages.

    Returns (system_msg, cleaned non-system turns).
    """
    system_msg = messages[0]
    turns = []
    for m in messages[1:]:
        if m["role"] == "assistant" and is_backchannel(m["content"]):
            continue
        if turns and turns[-1]["role"] == m["role"]:
            turns[-1] = {
                "role": m["role"],
                "content": turns[-1]["content"] + " " + m["content"],
            }
        else:
            turns.append({"role": m["role"], "content": m["content"]})
    return system_msg, turns


# ── Window sampling ───────────────────────────────────────────────────────────

def build_exchanges(turns: list[dict]) -> list[int]:
    """Indices i where (turns[i], turns[i+1]) is a (user, assistant) pair."""
    return [
        i for i in range(len(turns) - 1)
        if turns[i]["role"] == "user" and turns[i + 1]["role"] == "assistant"
    ]


def sample_windows(turns: list[dict], rng: random.Random) -> list[int]:
    """Pick non-overlapping window start-exchange indices for one conversation.

    A window covers exchanges [s, s + EXCHANGES_PER_WIN); its final assistant
    turn must have >= MIN_FINAL_WORDS words.
    """
    exchanges = build_exchanges(turns)
    n_windows = max(1, len(turns) // MSGS_PER_WINDOW)

    candidates = list(range(len(exchanges) - EXCHANGES_PER_WIN + 1))
    rng.shuffle(candidates)

    used: set[int] = set()
    starts = []
    for s in candidates:
        if len(starts) >= n_windows:
            break
        span = set(range(s, s + EXCHANGES_PER_WIN))
        if span & used:
            continue
        final_asst = turns[exchanges[s + EXCHANGES_PER_WIN - 1] + 1]
        if len(final_asst["content"].split()) < MIN_FINAL_WORDS:
            continue
        used |= span
        starts.append(s)
    return sorted(starts)


# ── Summaries (same method as chunk_conversation in self_play.py) ─────────────

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


def build_record(
    system_msg: dict,
    turns: list[dict],
    window_turns: list[dict],
    previous_turns: list[dict],
    ds_client: OpenAI,
) -> dict:
    """Assemble one condensed record, summarizing prior turns if any."""
    if previous_turns:
        summary = summarize_previous_turns(previous_turns, ds_client)
        sys_msg = {
            "role": "system",
            "content": (
                system_msg["content"]
                + "\n\n[Summary of conversation so far]\n"
                + summary
            ),
        }
    else:
        sys_msg = system_msg
    return {"messages": [sys_msg] + window_turns}


# ── Main pipeline ─────────────────────────────────────────────────────────────

def condense(input_path: Path, output_path: Path, seed: int) -> list[dict]:
    conversations = load_jsonl(input_path)
    rng = random.Random(seed)

    ds_client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )

    # Sample all windows first (deterministic, no API calls).
    jobs = []  # (key, system_msg, turns, window_turns, previous_turns)
    skipped = 0
    for conv_idx, conv in enumerate(conversations):
        system_msg, turns = clean_conversation(conv["messages"])
        exchanges = build_exchanges(turns)
        if len(exchanges) < EXCHANGES_PER_WIN:
            print(f"[condense] WARNING: conv {conv_idx} has "
                  f"{len(exchanges)} exchanges after cleaning — skipped")
            skipped += 1
            continue
        for s in sample_windows(turns, rng):
            lo = exchanges[s]
            hi = exchanges[s + EXCHANGES_PER_WIN - 1] + 2
            jobs.append((
                f"{conv_idx}:{lo}",
                system_msg,
                turns,
                turns[lo:hi],
                turns[:lo],
            ))

    print(f"[condense] Conversations : {len(conversations)} ({skipped} skipped)")
    print(f"[condense] Windows       : {len(jobs)} "
          f"({sum(1 for j in jobs if j[4])} need summaries)")

    # Resume support: skip windows already in the progress file.
    progress_path = output_path.with_suffix(".progress.jsonl")
    done: dict[str, dict] = {}
    if progress_path.exists():
        for rec in load_jsonl(progress_path):
            done[rec["key"]] = rec["record"]
        print(f"[condense] Resuming: {len(done)} windows already done")

    pending = [j for j in jobs if j[0] not in done]
    with progress_path.open("a") as progress_f, \
         ThreadPoolExecutor(max_workers=SUMMARY_WORKERS) as pool:

        def run_job(job):
            key, system_msg, turns, window_turns, previous_turns = job
            try:
                rec = build_record(system_msg, turns, window_turns,
                                   previous_turns, ds_client)
            except Exception as e:
                print(f"[condense] Retrying {key} after error: {e}")
                rec = build_record(system_msg, turns, window_turns,
                                   previous_turns, ds_client)
            return key, rec

        futures = [pool.submit(run_job, j) for j in pending]
        for n, fut in enumerate(as_completed(futures), 1):
            key, rec = fut.result()
            done[key] = rec
            progress_f.write(json.dumps({"key": key, "record": rec}) + "\n")
            progress_f.flush()
            if n % 25 == 0 or n == len(pending):
                print(f"[condense] {n}/{len(pending)} windows built")

    # Write final output in deterministic (conv, turn) order.
    ordered = [done[j[0]] for j in jobs]
    with output_path.open("w") as f:
        for rec in ordered:
            f.write(json.dumps(rec) + "\n")
    print(f"[condense] Wrote {len(ordered)} records → {output_path}")

    progress_path.unlink()
    return ordered


def print_stats(records: list[dict]) -> None:
    """Sanity stats: window counts, assistant word counts, token lengths."""
    asst = [m["content"] for r in records for m in r["messages"]
            if m["role"] == "assistant"]
    words = [len(a.split()) for a in asst]
    dist = Counter(min(w, 20) // 5 * 5 for w in words)
    print(f"[stats] Records                : {len(records)}")
    print(f"[stats] Assistant turns        : {len(asst)}")
    print(f"[stats] Assistant words min/med: {min(words)} / "
          f"{sorted(words)[len(words) // 2]}")
    for bucket in sorted(dist):
        label = f"{bucket}-{bucket + 4}" if bucket < 20 else "20+"
        print(f"[stats]   {label:>5} words: {dist[bucket]}")

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)
        lens = [
            len(tok.encode(tok.apply_chat_template(
                r["messages"], tokenize=False, add_generation_prompt=False)))
            for r in records
        ]
        print(f"[stats] Tokens min/median/max : {min(lens)} / "
              f"{sorted(lens)[len(lens) // 2]} / {max(lens)}")
        print(f"[stats] Records over 4096 tok : {sum(l > 4096 for l in lens)}")
    except Exception as e:
        print(f"[stats] Token check skipped: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build condensed SFT dataset")
    parser.add_argument("--input", required=True, help="Full-conversation JSONL")
    parser.add_argument("--output", required=True, help="Condensed output JSONL")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = condense(Path(args.input), Path(args.output), args.seed)
    print_stats(records)
