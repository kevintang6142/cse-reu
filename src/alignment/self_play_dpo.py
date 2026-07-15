"""
Branching self-play for turn-level DPO.

At every therapist decision point of a conversation:
  1. Keep the conversation history fixed and sample K candidate therapist turns
     at a HIGH temperature (diverse branches).
  2. Roll each candidate out to the end of the session at the normal (lower)
     self-play temperature, so rollout noise stays low and score differences are
     attributable to the candidate turn.
  3. Score every completed rollout with the CTRS-R critic (0-33 total).
  4. The candidate whose rollout scored highest is `chosen`, lowest is
     `rejected`. A pair is emitted only if the two candidates are distinct text
     and their rollout scores differ by >= min_margin. The prompt is the SHARED
     history, so each pair is exactly the same-context comparison DPO assumes.
  5. The conversation advances along the best branch: the winning candidate plus
     the patient reply from its rollout are appended, and the next therapist
     turn becomes the next decision point. If the best rollout ended right after
     the candidate, the conversation is over.

Cost: a conversation with T therapist turns generates ~T*K rollouts and critic
calls, roughly an order of magnitude more than plain self-play per vignette —
size n_vignettes accordingly.

Usage (standalone):
    python -m src.alignment.self_play_dpo --output-pairs dpo_pairs.jsonl \
        --output-transcripts transcripts.jsonl --n-vignettes 100
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

import torch
from openai import AsyncOpenAI

from src.alignment.self_play import (
    API_CONCURRENCY,
    BASE_MODEL_NAME,
    MAX_TURNS,
    MIN_TURNS,
    THERAPIST_BATCH_SIZE,
    THERAPIST_SYSTEM,
    USER_START_MSG,
    ensure_user_first,
    get_critic_feedback_raw,
    load_therapist_model,
    load_vignettes,
    parse_critic_json,
    patient_turn_async,
    should_end_session_async,
    therapist_turn_batch,
    to_hf_messages,
)
from src.alignment.kto import is_degenerate, total_ctsr_score


# ── Defaults ──────────────────────────────────────────────────────────────────
N_VIGNETTES_DPO    = 100
K_CANDIDATES       = 3
BRANCH_TEMPERATURE = 1.0   # turned up for diverse candidate turns
ROLLOUT_TEMPERATURE = 0.7  # back down for rollouts — matches plain self-play
MIN_SCORE_MARGIN   = 2     # chosen rollout must beat rejected by >= this (of 33)


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _n_therapist(turns: list[dict]) -> int:
    return sum(t["role"] == "therapist" for t in turns)


def _score_rollout(raw_critique: str) -> int | None:
    """CTRS-R total (0-33) of a rollout, or None if the critique is unusable."""
    return total_ctsr_score({"parsed": parse_critic_json(raw_critique)})


def select_pair(
    scores: list[int | None], texts: list[str], min_margin: int = MIN_SCORE_MARGIN,
) -> tuple[int, int] | None:
    """Pick (chosen_idx, rejected_idx) among a decision point's candidates.

    Requires two scored, non-empty, textually distinct candidates whose rollout
    scores differ by at least min_margin. Returns None if no valid pair exists.
    """
    valid = [i for i, (s, t) in enumerate(zip(scores, texts)) if s is not None and t.strip()]
    if len(valid) < 2:
        return None
    best = max(valid, key=lambda i: scores[i])
    worst = min(valid, key=lambda i: scores[i])
    if best == worst or scores[best] - scores[worst] < min_margin:
        return None
    if _norm_text(texts[best]) == _norm_text(texts[worst]):
        return None
    return best, worst


def _dpo_prompt_messages(history: list[dict]) -> list[dict]:
    """History → conversational DPO prompt (system + turns, guaranteed user turn).

    Mirrors kto._turn_records so DPO and KTO records tokenize identically.
    """
    messages = ensure_user_first(to_hf_messages(history, THERAPIST_SYSTEM))
    if len(messages) == 1:  # opening decision point: bare system prompt
        messages = messages + [USER_START_MSG]
    return messages


async def _finish_rollouts(
    model, tokenizer, client: AsyncOpenAI, sem: asyncio.Semaphore,
    rollouts: list[list[dict]], rollout_vignettes: list[dict],
    therapist_batch_size: int, temperature: float,
) -> None:
    """Continue every rollout (each ending on a therapist turn) in lockstep until
    the moderator ends its session or it reaches MAX_TURNS therapist turns.
    Mutates `rollouts` in place."""
    active = list(range(len(rollouts)))
    round_num = 0
    while active:
        round_num += 1
        # Moderator end-check, mirroring run_all_interviews eligibility.
        eligible = [i for i in active if _n_therapist(rollouts[i]) >= MIN_TURNS]
        if eligible:
            results = await asyncio.gather(*[
                should_end_session_async(client, sem, rollouts[i]) for i in eligible
            ])
            ended = {i for i, r in zip(eligible, results) if r}
            active = [i for i in active if i not in ended]
        active = [i for i in active if _n_therapist(rollouts[i]) < MAX_TURNS]
        if not active:
            break

        print(f"    rollout round {round_num}: patient×{len(active)}...", flush=True)
        replies = await asyncio.gather(*[
            patient_turn_async(client, sem, rollout_vignettes[i], rollouts[i])
            for i in active
        ])
        for i, msg in zip(active, replies):
            rollouts[i].append({"role": "patient", "content": msg})

        responses = therapist_turn_batch(
            model, tokenizer,
            [rollouts[i] for i in active],
            [THERAPIST_SYSTEM] * len(active),
            batch_size=therapist_batch_size,
            temperature=temperature,
        )
        for i, msg in zip(active, responses):
            rollouts[i].append({"role": "therapist", "content": msg})


async def _dpo_self_play(
    model, tokenizer, client: AsyncOpenAI, sem: asyncio.Semaphore,
    vignettes: list[dict],
    k_candidates: int,
    min_margin: int,
    therapist_batch_size: int,
    branch_temperature: float,
    rollout_temperature: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Run branching self-play over all vignettes in lockstep.

    Returns (pairs, transcripts, decisions):
      pairs       — TRL conversational DPO records {prompt, chosen, rejected, ...}
      transcripts — the main-line conversations (best branch at every point)
      decisions   — per-decision-point log of candidate rollout scores
    """
    n = len(vignettes)
    main_turns: list[list[dict]] = [[] for _ in range(n)]
    done = [False] * n
    pairs: list[dict] = []
    decisions: list[dict] = []

    decision_round = 0
    while True:
        active = [i for i in range(n) if not done[i]]
        if not active:
            break
        decision_round += 1
        print(f"\n{'='*60}")
        print(f"DECISION ROUND {decision_round}: {len(active)} active conversations, "
              f"{len(pairs)} pairs so far")
        print(f"{'='*60}")

        # ── 1) Branch: K candidates per conversation at high temperature ──────
        turns_batch = [main_turns[i] for i in active for _ in range(k_candidates)]
        sys_batch = [THERAPIST_SYSTEM] * len(turns_batch)
        print(f"  Branching: {len(turns_batch)} candidates "
              f"(K={k_candidates}, T={branch_temperature})...")
        cands = therapist_turn_batch(
            model, tokenizer, turns_batch, sys_batch,
            batch_size=therapist_batch_size, temperature=branch_temperature,
        )

        # ── 2) Build rollouts (dedupe identical candidates to save compute) ───
        rollouts: list[list[dict]] = []
        rollout_vignettes: list[dict] = []
        branch_slices: list[tuple[int, list[int]]] = []  # (conv idx, rollout idxs)
        for j, i in enumerate(active):
            raw_cands = cands[j * k_candidates:(j + 1) * k_candidates]
            uniq, seen = [], set()
            for c in raw_cands:
                key = _norm_text(c)
                if key and key not in seen:
                    seen.add(key)
                    uniq.append(c)
            if not uniq:  # all candidates empty — keep one so the line can advance
                uniq = [raw_cands[0]]
            idxs = []
            for c in uniq:
                idxs.append(len(rollouts))
                rollouts.append(list(main_turns[i]) + [{"role": "therapist", "content": c}])
                rollout_vignettes.append(vignettes[i])
            branch_slices.append((i, idxs))

        print(f"  Rolling out {len(rollouts)} branches to completion "
              f"(T={rollout_temperature})...")
        await _finish_rollouts(
            model, tokenizer, client, sem, rollouts, rollout_vignettes,
            therapist_batch_size, rollout_temperature,
        )

        # ── 3) Judge every rollout ─────────────────────────────────────────────
        print(f"  Scoring {len(rollouts)} rollouts with the CTRS-R critic...")
        raws = await asyncio.gather(*[
            get_critic_feedback_raw(client, sem, rollout_vignettes[r], rollouts[r])
            for r in range(len(rollouts))
        ])
        scores: list[int | None] = [_score_rollout(raw) for raw in raws]
        for r in range(len(rollouts)):
            # A candidate whose rollout collapses into repetition can never be
            # `chosen`, and becomes `rejected` against any scored sibling.
            if scores[r] is not None and is_degenerate(rollouts[r]):
                scores[r] = -1

        # ── 4) Select pairs and advance each conversation along its best branch ─
        n_new = 0
        for i, idxs in branch_slices:
            base_len = len(main_turns[i])
            texts = [rollouts[r][base_len]["content"] for r in idxs]
            sc = [scores[r] for r in idxs]

            picked = select_pair(sc, texts, min_margin)
            if picked is not None:
                b, w = picked
                pairs.append({
                    "prompt": _dpo_prompt_messages(main_turns[i]),
                    "chosen": [{"role": "assistant", "content": texts[b]}],
                    "rejected": [{"role": "assistant", "content": texts[w]}],
                    # Match self-play/KTO tokenization (thinking disabled).
                    "chat_template_kwargs": {"enable_thinking": False},
                    "meta": {
                        "vignette": vignettes[i]["name"],
                        "decision_index": _n_therapist(main_turns[i]),
                        "chosen_score": sc[b],
                        "rejected_score": sc[w],
                    },
                })
                n_new += 1
            decisions.append({
                "vignette": vignettes[i]["name"],
                "decision_index": _n_therapist(main_turns[i]),
                "n_candidates": len(idxs),
                "scores": sc,
                "paired": picked is not None,
            })

            # Advance along the best-scored branch (first candidate if none scored).
            valid = [k for k, (s, t) in enumerate(zip(sc, texts)) if s is not None and t.strip()]
            adv = max(valid, key=lambda k: sc[k]) if valid else 0
            adopted = rollouts[idxs[adv]]
            # Adopt candidate turn + the rollout's patient reply (if any), then
            # re-branch at the next therapist turn.
            main_turns[i] = adopted[:base_len + 2]
            if len(adopted) <= base_len + 1 or _n_therapist(main_turns[i]) >= MAX_TURNS:
                # Rollout ended right after the candidate, or the turn cap is hit.
                main_turns[i] = adopted[:base_len + 1]
                done[i] = True

        n_done = sum(done)
        print(f"  Round {decision_round} done: +{n_new} pairs "
              f"({len(pairs)} total), finished conversations: {n_done}/{n}")

    transcripts = [
        {"vignette": vignettes[i]["name"], "turns": main_turns[i]} for i in range(n)
    ]
    return pairs, transcripts, decisions


def run_self_play_dpo(
    output_pairs: str | Path,
    output_transcripts: str | Path | None = None,
    output_decisions: str | Path | None = None,
    base_model_name: str = BASE_MODEL_NAME,
    n_vignettes: int = N_VIGNETTES_DPO,
    vignette_seed: int | None = None,
    k_candidates: int = K_CANDIDATES,
    min_margin: int = MIN_SCORE_MARGIN,
    branch_temperature: float = BRANCH_TEMPERATURE,
    rollout_temperature: float = ROLLOUT_TEMPERATURE,
    api_concurrency: int = API_CONCURRENCY,
    therapist_batch_size: int = THERAPIST_BATCH_SIZE,
) -> Path:
    """Run branching self-play and save turn-level DPO pairs. Returns output_pairs."""
    output_pairs = Path(output_pairs)
    output_pairs.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Self-Play-DPO] Loading model: {base_model_name}")
    model, tokenizer = load_therapist_model(base_model_name)
    vignettes = load_vignettes(n_vignettes, seed=vignette_seed)

    client = AsyncOpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    sem = asyncio.Semaphore(api_concurrency)

    print(f"[Self-Play-DPO] Branching self-play: {len(vignettes)} vignettes, "
          f"K={k_candidates}, branch_T={branch_temperature}, rollout_T={rollout_temperature}, "
          f"min_margin={min_margin}")
    t0 = time.perf_counter()
    pairs, transcripts, decisions = asyncio.run(_dpo_self_play(
        model, tokenizer, client, sem, vignettes,
        k_candidates=k_candidates,
        min_margin=min_margin,
        therapist_batch_size=therapist_batch_size,
        branch_temperature=branch_temperature,
        rollout_temperature=rollout_temperature,
    ))
    elapsed = time.perf_counter() - t0
    n_decisions = len(decisions)
    print(f"[Self-Play-DPO] Done in {elapsed:.1f}s — {len(pairs)} pairs "
          f"from {n_decisions} decision points "
          f"({len(pairs) / max(n_decisions, 1):.0%} paired)")

    with open(output_pairs, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"[Self-Play-DPO] Pairs saved → {output_pairs}")

    if output_transcripts:
        output_transcripts = Path(output_transcripts)
        output_transcripts.parent.mkdir(parents=True, exist_ok=True)
        with open(output_transcripts, "w") as f:
            for t in transcripts:
                f.write(json.dumps(t) + "\n")
        print(f"[Self-Play-DPO] Main-line transcripts saved → {output_transcripts}")

    if output_decisions:
        output_decisions = Path(output_decisions)
        output_decisions.parent.mkdir(parents=True, exist_ok=True)
        with open(output_decisions, "w") as f:
            for d in decisions:
                f.write(json.dumps(d) + "\n")
        print(f"[Self-Play-DPO] Decision log saved → {output_decisions}")

    del model
    torch.cuda.empty_cache()
    return output_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Branching self-play for turn-level DPO")
    parser.add_argument("--output-pairs", required=True, help="Output DPO pairs JSONL")
    parser.add_argument("--output-transcripts", default=None,
                        help="Output main-line transcripts JSONL")
    parser.add_argument("--output-decisions", default=None,
                        help="Output per-decision-point score log JSONL")
    parser.add_argument("--base-model", default=BASE_MODEL_NAME)
    parser.add_argument("--n-vignettes", type=int, default=N_VIGNETTES_DPO)
    parser.add_argument("--vignette-seed", type=int, default=None)
    parser.add_argument("--k-candidates", type=int, default=K_CANDIDATES)
    parser.add_argument("--min-margin", type=int, default=MIN_SCORE_MARGIN)
    parser.add_argument("--branch-temperature", type=float, default=BRANCH_TEMPERATURE)
    parser.add_argument("--rollout-temperature", type=float, default=ROLLOUT_TEMPERATURE)
    parser.add_argument("--api-concurrency", type=int, default=API_CONCURRENCY)
    parser.add_argument("--therapist-batch-size", type=int, default=THERAPIST_BATCH_SIZE)
    args = parser.parse_args()

    run_self_play_dpo(
        output_pairs=args.output_pairs,
        output_transcripts=args.output_transcripts,
        output_decisions=args.output_decisions,
        base_model_name=args.base_model,
        n_vignettes=args.n_vignettes,
        vignette_seed=args.vignette_seed,
        k_candidates=args.k_candidates,
        min_margin=args.min_margin,
        branch_temperature=args.branch_temperature,
        rollout_temperature=args.rollout_temperature,
        api_concurrency=args.api_concurrency,
        therapist_batch_size=args.therapist_batch_size,
    )
