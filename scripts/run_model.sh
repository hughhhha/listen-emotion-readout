#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
size=${1:?Usage: bash scripts/run_model.sh 3B or 7B}
case "$size" in 3B|7B) ;; *) echo 'Expected 3B or 7B' >&2; exit 2 ;; esac
model=${MODEL_DIR:-models/Qwen2.5-Omni-$size}
data=${DATA_DIR:-data}
out=${RUN_DIR:-runs/$size}
export PYTHONHASHSEED=0
mkdir -p "$out"
for split in train test; do
    python scripts/infer_fast.py --manifest "$data/$split.jsonl" --audio-root "$data" \
        --model "$model" --output "$out/$split.jsonl" --batch-size 1 \
        --text-batch-size 1 --bucket-size 1 --stop-on-letter
done
cat "$out/train.jsonl" "$out/test.jsonl" > "$out/combined.jsonl.tmp"
mv "$out/combined.jsonl.tmp" "$out/combined.jsonl"
python scripts/methods.py --protocol official_plus4 --select-linear-head \
    --feature-names pitch_span,rms_span,pause_fraction --features "$data/features.jsonl" \
    --predictions "$out/combined.jsonl" --model "$model" --output "$out/methods" --jobs 4
python scripts/report_baseline.py --data "$data" --results "$out/train.jsonl" "$out/test.jsonl" \
    --heldout "$out/methods/heldout_predictions.jsonl" --output "$out/frozen_baseline.json"
