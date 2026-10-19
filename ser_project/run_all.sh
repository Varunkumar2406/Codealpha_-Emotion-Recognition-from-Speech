#!/bin/bash
# Train all three models on the same split and print a comparison.
# Usage: bash run_all.sh data/RAVDESS data/TESS [data/EMODB]
set -e
RAV=$1; TESS=$2; EMO=$3
ARGS="--ravdess $RAV --tess $TESS"; [ -n "$EMO" ] && ARGS="$ARGS --emodb $EMO"
for M in cnn lstm cnnlstm; do
  python -m ser.train $ARGS --model $M --epochs 60 --augment
done
python - <<'PY'
import json
print(f"\n{'model':10s} {'accuracy':>9s} {'macro-F1':>9s}")
for m in ["cnn","lstm","cnnlstm"]:
    r=json.load(open(f"outputs/{m}/metrics.json")); print(f"{m:10s} {r['accuracy']:9.3f} {r['macro_f1']:9.3f}")
PY
