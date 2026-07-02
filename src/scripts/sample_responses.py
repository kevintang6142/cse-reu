"""Sample a handful of random therapist (assistant) responses from the training set."""

import json
import random
from pathlib import Path

TRAIN_PATH = Path(__file__).parent.parent.parent / "data/processed/combined_train_chunked.jsonl"

records = []
with open(TRAIN_PATH) as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

# Collect all assistant turns
assistant_turns = []
for rec in records:
    for msg in rec["messages"]:
        if msg["role"] == "assistant":
            assistant_turns.append(msg["content"])

sampled = random.sample(assistant_turns, min(10, len(assistant_turns)))

print(f"Total assistant turns: {len(assistant_turns)}")
print(f"Sampled: {len(sampled)}\n")
print("=" * 70)

for i, text in enumerate(sampled, 1):
    print(f"\n[{i}] {text}\n")
    print("-" * 70)
