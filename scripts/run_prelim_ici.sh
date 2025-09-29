#!/usr/bin/env bash
set -euo pipefail

TOOLS="data/prelim/tools.jsonl"
TASKS="data/prelim/tasks.jsonl"
FEAT="data/prelim/tools.features_ici.jsonl"
OUTD="data/prelim/results"

mkdir -p "$(dirname "$FEAT")" "$OUTD"

echo "==> ICI feature extraction"
python3 ici/extract.py --tools "$TOOLS" --out "$FEAT.tmp"
python3 ici/score.py   --feat  "$FEAT.tmp" --out "$FEAT"
rm -f "$FEAT.tmp"

echo "==> Retrieval (λ=0.25, K=3)"
python3 -m retrieval.run_ir --tools "$TOOLS" --tasks "$TASKS" --ici "$FEAT" --lambda 0.25 --K 3 --outdir "$OUTD"

echo "==> Results:"
cat "$OUTD/preliminary_ir_results.csv"
