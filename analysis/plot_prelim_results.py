#!/usr/bin/env python3
"""Generate diagnostic plots for the preliminary ICI-aware retrieval study."""
import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt


def read_lambda_sweep(path: str) -> List[Dict[str, float]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "lambda": float(row["lambda"]),
                "Top-1": float(row["Top-1"]),
                "Recall": float(row[next(iter([c for c in row if c.startswith("Recall@")]))])
            })
    return rows


def plot_lambda_metrics(rows: List[Dict[str, float]], out_path: str) -> None:
    lambdas = [r["lambda"] for r in rows]
    top1 = [r["Top-1"] for r in rows]
    recall = [r["Recall"] for r in rows]

    plt.figure(figsize=(6, 4))
    plt.plot(lambdas, top1, marker="o", label="Top-1")
    plt.plot(lambdas, recall, marker="s", label="Recall@K")
    plt.xlabel("λ (ICI penalty strength)")
    plt.ylabel("Score")
    plt.title("Lambda Sweep: Text vs ICI-penalized")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def read_top1_analysis(path: str) -> List[Dict[str, float]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "mode": row["mode"],
                "lambda": float(row["lambda"]),
                "Top1_acc": float(row["Top1_acc"]),
                "Top1_mean_ICI": float(row["Top1_mean_ICI"]),
            })
    return rows


def plot_top1_analysis(rows: List[Dict[str, float]], out_path: str) -> None:
    modes = [r["mode"] for r in rows]
    accs = [r["Top1_acc"] for r in rows]
    mean_ici = [r["Top1_mean_ICI"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].bar(modes, accs, color=["#4C72B0", "#55A868"])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Top-1 Accuracy")
    axes[0].set_title("Top-1 Accuracy by Reranking Mode")

    axes[1].bar(modes, mean_ici, color=["#4C72B0", "#55A868"])
    axes[1].axhline(0, color="grey", linewidth=0.8)
    axes[1].set_ylabel("Mean ICI (z-score)")
    axes[1].set_title("Mean ICI of Top-1 Selection")

    for ax in axes:
        for tick in ax.get_xticklabels():
            tick.set_rotation(15)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def read_per_task(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def plot_capability_recall(rows, out_path: str) -> None:
    stats = defaultdict(lambda: {"count": 0, "text_hits": 0, "ici_hits": 0})
    for row in rows:
        cap = row["task_id"].split("-")[0]
        stats[cap]["count"] += 1
        if row["gold"] in row.get("rank_text", []):
            stats[cap]["text_hits"] += 1
        if row["gold"] in row.get("rank_ici", []):
            stats[cap]["ici_hits"] += 1

    caps = sorted(stats.keys())
    text_recalls = [stats[c]["text_hits"] / stats[c]["count"] for c in caps]
    ici_recalls = [stats[c]["ici_hits"] / stats[c]["count"] for c in caps]
    x = range(len(caps))
    width = 0.35

    plt.figure(figsize=(7, 4))
    plt.bar([i - width / 2 for i in x], text_recalls, width=width, label="Text-only")
    plt.bar([i + width / 2 for i in x], ici_recalls, width=width, label="ICI-penalized")
    plt.xticks(list(x), caps)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Recall@K")
    plt.title("Recall@K by Capability")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate plots for preliminary ICI-aware retrieval results.")
    parser.add_argument("--results", default="data/prelim/results", help="Directory containing result CSV/JSONL files.")
    parser.add_argument("--figdir", default="docs/figures", help="Output directory for generated figures.")
    parser.add_argument("--per-task", default="data/prelim/results/per_task_rankings.jsonl", help="Per-task rankings JSONL.")
    args = parser.parse_args()

    os.makedirs(args.figdir, exist_ok=True)

    lambda_rows = read_lambda_sweep(os.path.join(args.results, "lambda_sweep.csv"))
    plot_lambda_metrics(lambda_rows, os.path.join(args.figdir, "lambda_sweep.png"))

    top1_rows = read_top1_analysis(os.path.join(args.results, "top1_ici_analysis.csv"))
    plot_top1_analysis(top1_rows, os.path.join(args.figdir, "top1_ici.png"))

    per_task_rows = read_per_task(args.per_task)
    plot_capability_recall(per_task_rows, os.path.join(args.figdir, "capability_recall.png"))

    print(f"Saved plots to {args.figdir}")


if __name__ == "__main__":
    main()
