"""
Prepare CACTUS vignettes for self-play.

The CACTUS dataset (cactus-camel/cactus) contains ~31.5k rows, but the same
*person* (identified by an identical `intake_form`) appears in many rows — one per
CBT technique / attitude / automatic-thought variation. For self-play we only want
distinct people: a person's `patterns` (cognitive-distortion list) is constant across
all of their rows, so we collapse by `intake_form` and keep the patterns.

The distinct people are shuffled deterministically and split into:
  - test : the first N_TEST people (held out, never used for self-play)
  - train: the remaining people, from which each self-play iteration randomly
           samples N_VIGNETTES.

Outputs JSONL records of the form:
  {"id": "cactus_0001", "name": "Brooke Davis", "intake_form": "...", "patterns": [...]}

Usage:
    python -m src.data.prepare_cactus
    python -m src.data.prepare_cactus --n-test 100 --seed 42 --out-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

DATASET = "cactus-camel/cactus"
DEFAULT_OUT_DIR = Path("data/processed")
DEFAULT_N_TEST = 100
DEFAULT_SEED = 42


def _parse_name(intake_form: str) -> str:
    """Best-effort extraction of the person's name from the intake form header.

    The form starts with a 'Name:\\n<name>' block; fall back to empty string if the
    layout differs so a missing name never breaks preprocessing.
    """
    m = re.search(r"Name:\s*\n?\s*(.+)", intake_form)
    return m.group(1).strip() if m else ""


def collapse_to_people(ds) -> list[dict]:
    """Collapse dataset rows to one record per distinct person (unique intake_form)."""
    seen: dict[str, dict] = {}
    for row in ds:
        key = row["intake_form"].strip()
        if key in seen:
            continue
        seen[key] = {
            "intake_form": key,
            "patterns": list(row["patterns"]),
            "name": _parse_name(key),
        }
    # Stable, dataset-order list (insertion order) → deterministic before shuffling.
    return list(seen.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CACTUS vignettes for self-play")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-test", type=int, default=DEFAULT_N_TEST,
                        help="Number of distinct people held out for the test split")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Shuffle seed for the train/test split")
    args = parser.parse_args()

    print(f"[prepare_cactus] Loading {DATASET} …")
    ds = load_dataset(DATASET, split="train")
    print(f"[prepare_cactus] {len(ds)} raw rows")

    people = collapse_to_people(ds)
    n_people = len(people)
    print(f"[prepare_cactus] {n_people} distinct people (unique intake_form)")

    if args.n_test >= n_people:
        raise ValueError(f"--n-test ({args.n_test}) must be < distinct people ({n_people})")

    # Deterministic shuffle, then split.
    rng = random.Random(args.seed)
    rng.shuffle(people)
    for i, p in enumerate(people):
        p["id"] = f"cactus_{i:05d}"

    test_people = people[: args.n_test]
    train_people = people[args.n_test :]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "cactus_vignettes_train.jsonl"
    test_path = args.out_dir / "cactus_vignettes_test.jsonl"

    with train_path.open("w", encoding="utf-8") as f:
        for p in train_people:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with test_path.open("w", encoding="utf-8") as f:
        for p in test_people:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"[prepare_cactus] Train: {len(train_people)} people → {train_path}")
    print(f"[prepare_cactus] Test : {len(test_people)} people → {test_path}")


if __name__ == "__main__":
    main()
