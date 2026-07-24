"""
ENACT evaluation: re-score existing eval transcripts with the WHO ENACT rubric.

Adapts the ENhancing Assessment of Common Therapeutic factors (ENACT) v1.0
competency tool (reports/ENACT_inperson_published_220321.pdf, Kohrt et al. 2015)
for LLM-as-judge scoring of text-based sessions:

  - Item 1 (Non-verbal communication & active listening) is DROPPED — every one
    of its behaviours (eye contact, posture, nodding) is unobservable in text.
  - Items 2-15 are scored on the instrument's own 1-4 level scheme:
      Level 1 = any unhelpful/harmful behaviour observed
      Level 2 = no unhelpful behaviour; none or only some basic skills
      Level 3 = no unhelpful behaviour; ALL basic skills
      Level 4 = Level 3 PLUS any advanced skill
  - Behaviours only observable in person (e.g. facial expression, offering a
    seat) are treated as not applicable rather than counting against a level.
  - Total = sum of the 14 items, range 14-56.

Reads reports/<model>/ctsr_eval_transcripts.jsonl (produced by notebook 05) so
each model is judged on the SAME conversations as the CTRS-R eval, and writes
scores to reports/enact/<model>/enact_eval_scores.csv — a separate tree, so
nothing consumed by notebook 06 is touched.

Usage (from repo root):
    python -m src.scripts.enact_eval                # all DEFAULT_MODELS
    python -m src.scripts.enact_eval --models base  # subset
"""

from dotenv import load_dotenv

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

REPORTS_DIR = REPO_ROOT / "reports"
OUTPUT_ROOT = REPORTS_DIR / "enact"
VIGNETTES_PATH = REPO_ROOT / "data" / "processed" / "cactus_vignettes_test.jsonl"

JUDGE_MODEL = "deepseek-v4-flash"
JUDGE_MAX_TOKENS = 50000
API_CONCURRENCY = 500
PROGRESS_EVERY = 20

DEFAULT_MODELS = [
    "base",
    "Qwen3.5-4B-SFT-KTO-CACTUS-iter_01-merged",
    "Qwen3.5-4B-SFT-KTO-CACTUS-iter_02-merged",
    "Qwen3.5-4B-SFT-DPO-CACTUS-iter_01-merged",
    "Qwen3.5-4B-SFT-DPO-CACTUS-iter_02-merged",
]

# ── ENACT rubric, items 2-15 (item 1 is non-verbal-only; see module docstring) ─
# Behaviour lists transcribed from ENACT v1.0 (WHO 2021, CC BY-NC-SA).
ENACT_ITEMS = [
    {
        "number": 2, "name": "Verbal communication skills",
        "key": "item_02_verbal_communication", "short_label": "Verbal\nComm.",
        "harmful": [
            "Interrupts client",
            "Asks many suggestive or leading closed-ended questions (e.g., You didn't really want to do that, right?)",
            "Corrects client (what you really mean...) or uses accusatory statements (you shouldn't have said that to your husband)",
            "Uses culturally and age-inappropriate language and terms",
        ],
        "basic": [
            "Open ended questions",
            "Summarizing or paraphrasing statements",
            "Allows client to complete statements before responding",
        ],
        "advanced": [
            "Encourages client to continue explaining (tell me more about...)",
            "Uses clarifying statements in first person (I heard you say..., I understood...)",
            "Matches rhythm to clients, allowing longer or shorter pauses based on client",
        ],
    },
    {
        "number": 3, "name": "Explanation & promotion of confidentiality",
        "key": "item_03_confidentiality", "short_label": "Confident-\niality",
        "harmful": [
            "Forces client to disclose to helper or others",
            "Describes confidentiality inaccurately (e.g., I will only tell your family)",
            "Promises all things will be kept confidential without exceptions",
            "Minimizes client's concerns about confidentiality (e.g., it doesn't matter if anyone else hears us)",
        ],
        "basic": [
            "Explains concept of confidentiality",
            "Lists exceptions for breaking confidentiality for self-harm or harm to others",
            "Explains why it can be important to break confidentiality",
        ],
        "advanced": [
            "Details the referral process related to confidentiality and exceptions",
            "Asks questions to assess client's understanding of confidentiality",
            "Topics of discussion are appropriate to confidentiality of setting",
        ],
    },
    {
        "number": 4, "name": "Rapport building & self-disclosure",
        "key": "item_04_rapport", "short_label": "Rapport",
        "harmful": [
            "Dominates session describing a personal experience",
            "Minimizes client's problems by describing how the helper has dealt with this",
            "Asking unnecessary embarrassing personal questions",
            "Discusses confidential information of other clients",
        ],
        "basic": [
            "Introduces self and explains role",
            "Makes casual, informal conversation",
            "Asks for client's introduction, e.g., what client prefers to be called",
            "Shares general experience to relate to the client (e.g., about one's community/region)",
        ],
        "advanced": [
            "Asks for client's reflection related to helper's information that is shared",
            "Checks with client that they are comfortable (e.g., offer seat, preferred language)",
        ],
    },
    {
        "number": 5, "name": "Exploration & normalisation of feelings",
        "key": "item_05_feelings_exploration", "short_label": "Feelings\nExplor.",
        "harmful": [
            "Makes statements that client's response is unusual or atypical for others in similar situations (e.g., people don't usually react this way)",
            "Minimizes or dismisses client's feelings or emotions",
            "Forces client to describe emotions",
        ],
        "basic": [
            "Appropriately encourages client to share feelings",
            "Explains that others may share similar symptoms, reactions, and concerns, given similar experiences",
            "Asks client to reflect on the experience of sharing emotions",
        ],
        "advanced": [
            "Explores potential reasons for hesitance to share emotions",
            "Comments thoughtfully on client's expressed emotion to encourage emotional expression",
            "Validates emotional responses while also reframing potential harmful emotional reactions",
        ],
    },
    {
        "number": 6, "name": "Demonstration of empathy, warmth & genuineness",
        "key": "item_06_empathy", "short_label": "Empathy",
        "harmful": [
            "Critical of client's concerns",
            "Dismissive of client's concerns",
            "Helper's emotional response appears inappropriate, fake or acting",
        ],
        "basic": [
            "Is warm, friendly, and genuine throughout session",
            "Continuously shows concern or care for the client (e.g., That sounds sad, can you tell me more about it?)",
            "Asks question to identify what emotions the client was feeling (e.g., I wonder if you felt sad or angry when this happened)",
        ],
        "advanced": [
            "Asks client to reflect on empathic statements from helper (e.g., What did you think when I said you sounded sad?)",
        ],
    },
    {
        "number": 7, "name": "Assessment of harm & collaborative response plan",
        "key": "item_07_harm_assessment", "short_label": "Harm\nAssess.",
        "harmful": [
            "Does not ask about self-harm",
            "Lectures client with religious or legal reasons against self-harm (e.g., this is sin, or this is against the law)",
            "Expresses disbelief (e.g., accuses client of discussing self-harm to get attention; states that others would not actually harm the client or client's children)",
            "Encourages client not to tell anyone else about self-harm or harm to others",
        ],
        "basic": [
            "Asks about self-harm or harm to others, or explores harm if raised by client",
            "Asks about current intent, means, or prior attempts",
            "Asks about risk and/or protective factors",
        ],
        "advanced": [
            "If current risk is high or low, helps client to develop safety plan (e.g., coping strategies and help seeking)",
        ],
    },
    {
        "number": 8, "name": "Connection to social functioning & impact on life",
        "key": "item_08_social_functioning", "short_label": "Social\nFunct.",
        "harmful": [
            "Criticizes client for letting symptoms impact functioning (e.g., you are weak, you have no willpower)",
            "Tells client there is no connection between mental health concerns and daily functioning, or does not ask how mental health is affecting daily functioning",
            "Criticizes client for impact of their problems on children, spouse, or family members",
            "Makes client feel guilty for impact on children, family, and others",
        ],
        "basic": [
            "Asks about daily functioning",
            "Discusses the connection (the relationship) between daily functioning and mental health",
        ],
        "advanced": [
            "Clarifies and/or supports client's connections between functioning and mental health or reframes as needed",
            "Explores relationship in both directions (daily life to symptoms; symptoms to daily life)",
            "Asks about history of daily functioning compared to current social context (e.g., how long has this been going on?)",
        ],
    },
    {
        "number": 9, "name": "Exploration of explanation for problem (causal & explanatory models)",
        "key": "item_09_explanatory_models", "short_label": "Explan.\nModels",
        "harmful": [
            "Criticizes client's view of problem as ignorant, superstitious, etc.",
            "Endorses harmful beliefs of client or social network",
        ],
        "basic": [
            "Asks about client's view on cause of problem",
            "Asks about family's or social support network's view on cause of problem (e.g., What does your family say caused this?)",
        ],
        "advanced": [
            "Incorporates client's perspective of cause in care planning in non-harmful manner",
            "Discusses alternative to harmful explanations",
            "Addresses differences in client's view of cause and others' view of cause",
        ],
    },
    {
        "number": 10, "name": "Appropriate involvement of family members & other close persons",
        "key": "item_10_family_involvement", "short_label": "Family\nInvolv.",
        "harmful": [
            "Tells client not to involve family or close person in any way during treatment or recovery",
            "Forces client to involve family or close person in treatment process",
            "Demands to speak with family or close person without permission from client",
            "Allows an accompanying close person to disempower the client",
        ],
        "basic": [
            "Asks about close person(s) in client's life (e.g., household members, family, or other)",
            "Asks client how they would like to involve close person(s) in the care process",
            "Asks client who they live with",
        ],
        "advanced": [
            "Explores client's choices or reasons for involving or not involving close, familiar person",
            "Does role-play or discusses options for successful interaction with close person",
        ],
    },
    {
        "number": 11, "name": "Collaborative goal setting & addressing client's expectations",
        "key": "item_11_goal_setting", "short_label": "Goal\nSetting",
        "harmful": [
            "Tells client that their goals (expectations) can't be met but does not give a reason",
            "Gives incorrect, misleading, or unrealistic information about treatment goals",
            "Dictates goal for client (forces goal upon client)",
        ],
        "basic": [
            "Asks client about goals (expectations)",
            "Clearly explains how client's goals and expectations fit with treatment plan",
        ],
        "advanced": [
            "Prioritizing and modification of treatment plan to fit client goals (expectations)",
            "Works with client to reframe their goals within scope of the treatment plan",
        ],
    },
    {
        "number": 12, "name": "Promotion of realistic hope for change",
        "key": "item_12_hope", "short_label": "Hope",
        "harmful": [
            "Makes negative statements about client's doubts (how do you expect to get better if you have no hope...)",
            "Gives unrealistic expectations (everything will be cured or solved...)",
            "Provides no hope for change (this problem cannot be solved...)",
        ],
        "basic": [
            "Explains how client can be hopeful about possibility of change",
            "Praises client for seeking care",
        ],
        "advanced": [
            "Solicits and explores client's doubts about the treatment",
            "Helper shares reasons for hope based on helper's prior experience or client's behaviours",
            "Discusses reasons for hope when client is doubtful or dissatisfied",
        ],
    },
    {
        "number": 13, "name": "Incorporation of coping mechanisms & prior solutions",
        "key": "item_13_coping", "short_label": "Coping",
        "harmful": [
            "Makes negative statements about client's coping mechanisms (that would never work...)",
            "Encourages or shows acceptance of harmful coping mechanisms",
        ],
        "basic": [
            "Asks client about current or past coping mechanisms (how they keep going after the problem started...)",
            "Praises client for positive or safe current or prior solutions",
        ],
        "advanced": [
            "Encourages use of continued positive coping mechanisms",
            "Reflection on prior unhealthy strategies and brainstorm positive alternatives",
        ],
    },
    {
        "number": 14, "name": "Psychoeducation & use of local terminology",
        "key": "item_14_psychoeducation", "short_label": "Psycho-\neducation",
        "harmful": [
            "Uses technical terms without checking client's understanding",
            "Uses stigmatizing mental health terms",
        ],
        "basic": [
            "Conducts accurate psychoeducation using simple terms",
            "Includes local concepts and terminology into psychoeducation",
        ],
        "advanced": [
            "Incorporates client's description of the problem",
            "Checks that client understands psychoeducation",
        ],
    },
    {
        "number": 15, "name": "Elicitation of feedback on advice, suggestions & recommendations",
        "key": "item_15_feedback_elicitation", "short_label": "Feedback\nElicit.",
        "harmful": [
            "Lectures client about what to do without asking for client feedback",
            "Offers negative or harmful suggestions",
        ],
        "basic": [
            "Asks for feedback from client to see if any offered suggestions are helpful",
            "Provides clarifications, reframing, or alternative suggestions based on feedback",
        ],
        "advanced": [
            "Summarizes feedback provided by client and checks if interpretation is correct",
        ],
    },
]

ITEM_KEYS = [item["key"] for item in ENACT_ITEMS]
ITEM_LABELS = [item["short_label"] for item in ENACT_ITEMS]

SEM = asyncio.Semaphore(API_CONCURRENCY)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_judge_prompt(intake_form: str, turns: list[dict]) -> str:
    """Build an ENACT judge prompt that scores items 2-15 in one call."""
    transcript_text = "\n".join(
        f"{'Therapist' if t['role'] == 'therapist' else 'Patient'}: {t['content']}"
        for t in turns
    )

    rubric_text = ""
    for item in ENACT_ITEMS:
        harmful = "\n".join(f"      - {b}" for b in item["harmful"])
        basic = "\n".join(f"      - {b}" for b in item["basic"])
        advanced = "\n".join(f"      - {b}" for b in item["advanced"])
        rubric_text += (
            f"  Item {item['number']}: {item['name']} (key: {item['key']})\n"
            f"    Unhelpful or potentially harmful behaviours:\n{harmful}\n"
            f"    Basic helping skills:\n{basic}\n"
            f"    Advanced helping skills:\n{advanced}\n\n"
        )

    json_lines = ",\n".join(
        f'  "{item["key"]}": {{"reasoning": "...", "level": N}}' for item in ENACT_ITEMS
    )

    return (
        "You are a clinical supervisor scoring a text-based counselling session "
        "using the ENhancing Assessment of Common Therapeutic factors (ENACT) "
        "competency tool (WHO, v1.0).\n\n"
        "For EACH item below, first check which behaviours from each category the "
        "therapist demonstrated anywhere in the session, then assign exactly one "
        "level using this rule:\n"
        "Level 1 = the therapist showed ANY unhelpful or potentially harmful behaviour for this item "
        "(this overrides everything else).\n"
        "Level 2 = no unhelpful behaviour, and none or only some (not all) of the basic helping skills.\n"
        "Level 3 = no unhelpful behaviour, and ALL of the basic helping skills.\n"
        "Level 4 = Level 3 PLUS at least one advanced helping skill.\n\n"
        "This is a TEXT-based session: behaviours only observable in person "
        "(eye contact, posture, facial expression, offering a seat) are NOT "
        "applicable — do not count them for or against any level; judge each "
        "category on the behaviours that can be evidenced in text.\n\n"
        f"Patient background (intake form):\n{intake_form}\n\n"
        f"Transcript:\n{transcript_text}\n\n"
        f"ENACT Rubric (items 2-15; item 1 is non-verbal and excluded):\n{rubric_text}\n"
        "Score the therapist on ALL 14 items. For each item, provide your "
        "reasoning with specific transcript references (which harmful behaviours "
        "occurred, which basic skills were/weren't shown, which advanced skills "
        "were shown), then assign the level.\n\n"
        "Output ONLY a JSON object with this exact structure:\n"
        "{\n"
        f"{json_lines}\n"
        "}\n\n"
        "Rules:\n"
        "- Output ONLY the JSON object, no other text.\n"
        "- Each level must be an integer 1-4.\n"
        "- Each reasoning must be concise (2-4 sentences) with specific transcript references.\n"
        "- Do NOT use backslash-escaped single quotes in strings. Use plain apostrophes."
    )


def _sanitize_json(text: str) -> str:
    return text.replace("\\'", "'")


async def score_transcript_async(
    client: AsyncOpenAI, intake_form: str, turns: list[dict],
) -> dict:
    """Score all 14 ENACT items for a single transcript in one API call."""
    prompt = build_judge_prompt(intake_form, turns)
    async with SEM:
        resp = await client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=JUDGE_MAX_TOKENS,
            temperature=0.0,
            extra_body={"thinking": {"type": "disabled"}},
        )
    raw = resp.choices[0].message.content.strip()

    cleaned = raw
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = _sanitize_json(cleaned)

    results = {}
    try:
        data = json.loads(cleaned)
        for key in ITEM_KEYS:
            if key in data and isinstance(data[key], dict):
                level = int(data[key].get("level", 0))
                level = max(1, min(4, level))
                reasoning = data[key].get("reasoning", "")
            else:
                level = float("nan")
                reasoning = f"MISSING_KEY: {key}"
            results[key] = level
            results[key + "_rationale"] = reasoning
    except (json.JSONDecodeError, ValueError):
        for key in ITEM_KEYS:
            pattern = rf'"{key}".*?"level"\s*:\s*(\d)'
            m = re.search(pattern, cleaned)
            if m:
                results[key] = max(1, min(4, int(m.group(1))))
                results[key + "_rationale"] = f"PARTIAL_PARSE: {cleaned[:200]}"
            else:
                results[key] = float("nan")
                results[key + "_rationale"] = f"PARSE_ERROR: {raw[:200]}"
    return results


async def score_model(client: AsyncOpenAI, model_name: str, vignette_map: dict) -> pd.DataFrame:
    """Score every transcript of one model; returns the scores DataFrame."""
    transcripts_path = REPORTS_DIR / model_name / "ctsr_eval_transcripts.jsonl"
    transcripts = load_jsonl(transcripts_path)
    print(f"[ENACT] {model_name}: {len(transcripts)} transcripts from {transcripts_path}", flush=True)

    done = 0
    lock = asyncio.Lock()

    async def one(t: dict) -> dict:
        nonlocal done
        intake = vignette_map[t["vignette"]]["intake_form"]
        result = await score_transcript_async(client, intake, t["turns"])
        async with lock:
            done += 1
            if done % PROGRESS_EVERY == 0 or done == len(transcripts):
                print(f"[ENACT]   {done}/{len(transcripts)} judged", flush=True)
        return result

    results = await asyncio.gather(*(one(t) for t in transcripts))

    rows = []
    for t, result in zip(transcripts, results):
        row = {"model": t["model"], "vignette": t["vignette"]}
        row.update(result)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("vignette").reset_index(drop=True)


async def run(models: list[str]) -> None:
    vignettes = load_jsonl(VIGNETTES_PATH)
    vignette_map = {v["id"]: v for v in vignettes}

    client = AsyncOpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )

    for model_name in models:
        out_dir = OUTPUT_ROOT / model_name
        out_csv = out_dir / "enact_eval_scores.csv"
        if out_csv.exists():
            print(f"[ENACT] {model_name}: {out_csv} already exists — skipping "
                  f"(delete it to re-score)", flush=True)
            continue

        t0 = time.perf_counter()
        df = await score_model(client, model_name, vignette_map)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

        totals = df[ITEM_KEYS].sum(axis=1, min_count=len(ITEM_KEYS))
        n_parsed = totals.notna().sum()
        print(f"[ENACT] {model_name}: done in {time.perf_counter() - t0:.1f}s — "
              f"{n_parsed}/{len(df)} fully parsed, "
              f"mean total {totals.mean():.2f}/56, mean item {df[ITEM_KEYS].mean().mean():.2f}/4",
              flush=True)
        print(f"[ENACT] Scores → {out_csv}", flush=True)

    print("[ENACT] All models done.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score eval transcripts with the ENACT rubric")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="reports/<model> folders to score (default: base + KTO/DPO iters 1-2)")
    args = parser.parse_args()
    asyncio.run(run(args.models))


if __name__ == "__main__":
    main()
