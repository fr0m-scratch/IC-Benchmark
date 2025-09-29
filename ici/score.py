#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute ICI from extracted features with fixed weights (no learning).
Input : --feat data/prelim/tools.features_ici.jsonl
Output: --out  data/prelim/tools.features_ici.jsonl (overwritten or new)
"""
import argparse, json, statistics

WEIGHTS = {
    "depth": 0.20,
    "branching": 0.20,
    "num_params": 0.15,
    "num_required": 0.10,
    "constraints": 0.10,
    "enum_avg": 0.10,
    "desc_tokens": 0.10,
    "name_readability": 0.05
}
ORDER = list(WEIGHTS.keys())

def zscores(vals):
    m = statistics.mean(vals)
    sd = statistics.pstdev(vals) or 1.0
    return [(v - m)/sd for v in vals]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(x) for x in open(args.feat, "r", encoding="utf-8") if x.strip()]
    # z-normalize per feature
    zmap = {}
    for k in ORDER:
        zmap[k] = zscores([r[k] for r in rows])
    # compute ICI
    out_rows = []
    for idx, r in enumerate(rows):
        ici = sum(WEIGHTS[k] * zmap[k][idx] for k in ORDER)
        r["ici"] = ici
        out_rows.append(r)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
