#!/bin/bash
# One-shot resume after the iter_01 DPO-training OOM (2026-07-19):
# re-runs DPO training with the fixed memory settings on the already-generated
# pairs, merges the adapter, then continues the SFT+DPO loop at iteration 2.
set -e
cd /home/kevint/cse-reu
PY=/data_1_8TB_ssd/kevint/.venv/bin/python3
M=/data_1_8TB_ssd/kevint/models
export PYTHONUNBUFFERED=1

echo "=== RESUME: iter_01 DPO training (post-OOM fix: batch 1 x accum 16) ==="
$PY -m src.alignment.dpo \
  --train-data data/self_play_sft_dpo/iter_01/dpo/dpo_train.jsonl \
  --val-data data/self_play_sft_dpo/iter_01/dpo/dpo_val.jsonl \
  --output-dir "$M/Qwen3.5-4B-SFT-DPO-CACTUS-iter_01-dpo" \
  --model-name "$M/Qwen3.5-4B-SFT-DPO-CACTUS-iter_01-sftmerged"

echo "=== RESUME: merging iter_01 DPO adapter ==="
$PY - "$M" <<'PYEOF'
import sys
sys.path.insert(0, ".")
from src.alignment.kto import merge_adapter
m = sys.argv[1]
merge_adapter(
    f"{m}/Qwen3.5-4B-SFT-DPO-CACTUS-iter_01-sftmerged",
    f"{m}/Qwen3.5-4B-SFT-DPO-CACTUS-iter_01-dpo",
    f"{m}/Qwen3.5-4B-SFT-DPO-CACTUS-iter_01-merged",
)
PYEOF

echo "=== RESUME: iteration 1 complete, starting iteration 2 ==="
$PY -m src.alignment.train_loop_sft_dpo --n-iters 2 --start-iter 2 --start-from-adapter
