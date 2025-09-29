#!/usr/bin/env python3
"""Analyze SLM vs LLM gaps across ICI buckets for text vs ICI reranking."""
import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt


def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_ici_map(features_path: str) -> Dict[str, float]:
    return {row["tool_id"]: row["ici"] for row in load_jsonl(features_path)}


def compute_cutoffs(values: List[float]) -> List[float]:
    if len(set(values)) <= 1:
        return []
    return statistics.quantiles(values, n=5, method="inclusive")


def assign_bucket(value: float, cutoffs: List[float]) -> str:
    if not cutoffs:
        return "Q1"
    for idx, cutoff in enumerate(cutoffs, start=1):
        if value <= cutoff:
            return f"Q{idx}"
    return f"Q{len(cutoffs)+1}"


def error_category(row: Dict) -> str:
    if row.get("pass_at_1"):
        return "pass"
    err = row.get("error_type") or ""
    if err.startswith("schema_error"):
        return "schema_error"
    if err in {"empty_response", "no_json_found", "json_decode_error", "missing_tool_id", "arguments_not_object"}:
        return "format_error"
    if err and err.startswith("generation_error"):
        return "generation_error"
    if not row.get("selection_correct"):
        return "selection_error"
    if row.get("selection_correct") and not row.get("schema_valid"):
        return "schema_error"
    return "other_error"


def aggregate(rows: List[Dict]) -> Dict[str, float]:
    n = len(rows) or 1
    return {
        "count": len(rows),
        "pass@1": sum(1.0 for r in rows if r.get("pass_at_1")) / n,
        "selection_accuracy": sum(1.0 for r in rows if r.get("selection_correct")) / n,
        "json_valid_rate": sum(1.0 for r in rows if r.get("json_valid")) / n,
        "schema_valid_rate": sum(1.0 for r in rows if r.get("schema_valid")) / n,
    }


def bucketize(rows: List[Dict], cutoffs: List[float]) -> Dict[str, List[Dict]]:
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        buckets[row["ici_bucket"]].append(row)
    # ensure empty buckets still appear
    for idx in range(1, len(cutoffs) + 2 if cutoffs else 2):
        buckets.setdefault(f"Q{idx}", [])
    return buckets


def save_csv(path: str, rows: List[Dict], fieldnames: List[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_success_vs_ici(buckets_order: List[str],
                        slm_data: Dict[str, Dict[str, float]],
                        llm_data: Dict[str, Dict[str, float]],
                        title: str,
                        out_path: str):
    xs = range(len(buckets_order))
    slm_vals = [slm_data[b]["pass@1"] for b in buckets_order]
    llm_vals = [llm_data[b]["pass@1"] for b in buckets_order]
    plt.figure(figsize=(6, 4))
    plt.plot(xs, slm_vals, marker="o", label="SLM")
    plt.plot(xs, llm_vals, marker="s", label="LLM")
    plt.xticks(xs, buckets_order)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Pass@1")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_gap_curve(buckets_order: List[str],
                   gap_before: Dict[str, float],
                   gap_after: Dict[str, float],
                   out_path: str):
    xs = range(len(buckets_order))
    before = [gap_before[b] for b in buckets_order]
    after = [gap_after[b] for b in buckets_order]
    plt.figure(figsize=(6, 4))
    plt.plot(xs, before, marker="o", label="Gap (text-only)")
    plt.plot(xs, after, marker="s", label="Gap (ICI-aware)")
    plt.xticks(xs, buckets_order)
    plt.axhline(0.0, color="gray", linewidth=0.8)
    plt.ylabel("LLM − SLM Pass@1")
    plt.title("Gap vs ICI quantile")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_error_stacks(buckets_order: List[str],
                       slm_errors: Dict[str, Dict[str, float]],
                       llm_errors: Dict[str, Dict[str, float]],
                       out_path: str):
    categories = ["selection_error", "schema_error", "format_error", "generation_error", "other_error"]
    width = 0.35
    xs = range(len(buckets_order))

    def stack_values(error_map):
        vals = []
        for bucket in buckets_order:
            bucket_vals = []
            remaining = 1.0
            for cat in categories:
                bucket_vals.append(error_map[bucket].get(cat, 0.0))
                remaining -= bucket_vals[-1]
            bucket_vals.append(max(0.0, remaining))  # pass or residual
            vals.append(bucket_vals)
        return vals

    slm_vals = stack_values(slm_errors)
    llm_vals = stack_values(llm_errors)
    cat_labels = categories + ["pass"]

    plt.figure(figsize=(7, 4))
    bottom = [0.0] * len(buckets_order)
    for idx, cat in enumerate(cat_labels):
        slm_bar = [vals[idx] for vals in slm_vals]
        plt.bar([x - width / 2 for x in xs], slm_bar, width=width, bottom=bottom, label=f"SLM {cat}" if idx == 0 else None)
        bottom = [b + v for b, v in zip(bottom, slm_bar)]

    bottom = [0.0] * len(buckets_order)
    for idx, cat in enumerate(cat_labels):
        llm_bar = [vals[idx] for vals in llm_vals]
        plt.bar([x + width / 2 for x in xs], llm_bar, width=width, bottom=bottom, label=f"LLM {cat}" if idx == 0 else None, alpha=0.7)
        bottom = [b + v for b, v in zip(bottom, llm_bar)]

    plt.xticks(xs, buckets_order)
    plt.ylabel("Proportion")
    plt.ylim(0.0, 1.05)
    plt.title("Error decomposition by ICI bucket")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def error_distribution(rows: List[Dict]) -> Dict[str, float]:
    if not rows:
        return {}
    counts = defaultdict(int)
    for row in rows:
        counts[error_category(row)] += 1
    total = len(rows)
    return {k: v / total for k, v in counts.items()}


def main():
    parser = argparse.ArgumentParser(description="Analyze SLM vs LLM gap across ICI buckets")
    parser.add_argument("--slm-text", required=True)
    parser.add_argument("--llm-text", required=True)
    parser.add_argument("--slm-ici", required=True)
    parser.add_argument("--llm-ici", required=True)
    parser.add_argument("--ici-features", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--figdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.figdir, exist_ok=True)

    ici_map = build_ici_map(args.ici_features)

    datasets = {
        "slm_text": load_jsonl(args.slm_text),
        "llm_text": load_jsonl(args.llm_text),
        "slm_ici": load_jsonl(args.slm_ici),
        "llm_ici": load_jsonl(args.llm_ici),
    }

    # Derive cutoffs from all gold tools appearing in runs
    ici_values = []
    for rows in datasets.values():
        for row in rows:
            gold = row.get("gold")
            if gold in ici_map:
                row["ici_gold"] = ici_map[gold]
                ici_values.append(ici_map[gold])
            else:
                row["ici_gold"] = math.nan
    cutoffs = compute_cutoffs(sorted(ici_values))

    for rows in datasets.values():
        for row in rows:
            val = row.get("ici_gold")
            if val is None or math.isnan(val):
                row["ici_bucket"] = "Q1"
            else:
                row["ici_bucket"] = assign_bucket(val, cutoffs)

    buckets_order = [f"Q{i}" for i in range(1, len(cutoffs) + 2 if cutoffs else 2)]

    bucket_metrics = []
    aggregated = {}
    bucketized = {}
    for name, rows in datasets.items():
        buckets = bucketize(rows, cutoffs)
        bucketized[name] = buckets
        metrics = {bucket: aggregate(rows) for bucket, rows in buckets.items()}
        aggregated[name] = metrics
        for bucket in buckets_order:
            entry = {
                "run": name,
                "bucket": bucket,
                "pass@1": round(metrics[bucket]["pass@1"], 3),
                "selection_accuracy": round(metrics[bucket]["selection_accuracy"], 3),
                "json_valid_rate": round(metrics[bucket]["json_valid_rate"], 3),
            }
            bucket_metrics.append(entry)

    save_csv(
        os.path.join(args.outdir, "bucket_metrics.csv"),
        bucket_metrics,
        ["run", "bucket", "pass@1", "selection_accuracy", "json_valid_rate"],
    )

    # Overall summary
    def overall(rows):
        return aggregate(rows)["pass@1"], aggregate(rows)["selection_accuracy"], aggregate(rows)["json_valid_rate"]

    summary_rows = []
    for name, rows in datasets.items():
        metrics = aggregate(rows)
        summary_rows.append({
            "run": name,
            "pass@1": round(metrics["pass@1"], 3),
            "selection_accuracy": round(metrics["selection_accuracy"], 3),
            "json_valid_rate": round(metrics["json_valid_rate"], 3),
        })

    save_csv(
        os.path.join(args.outdir, "summary_overall.csv"),
        summary_rows,
        ["run", "pass@1", "selection_accuracy", "json_valid_rate"],
    )

    # Gap calculations per bucket
    gap_rows = []
    gap_before = {}
    gap_after = {}
    for bucket in buckets_order:
        gap_before[bucket] = aggregated["llm_text"][bucket]["pass@1"] - aggregated["slm_text"][bucket]["pass@1"]
        gap_after[bucket] = aggregated["llm_ici"][bucket]["pass@1"] - aggregated["slm_ici"][bucket]["pass@1"]
        gap_rows.append({
            "bucket": bucket,
            "gap_text": round(gap_before[bucket], 3),
            "gap_ici": round(gap_after[bucket], 3),
            "delta_gap": round(gap_after[bucket] - gap_before[bucket], 3),
        })

    save_csv(os.path.join(args.outdir, "gap_by_bucket.csv"), gap_rows, ["bucket", "gap_text", "gap_ici", "delta_gap"])

    # Overall deltas
    pass_slm_text = aggregate(datasets["slm_text"])["pass@1"]
    pass_slm_ici = aggregate(datasets["slm_ici"])["pass@1"]
    pass_llm_text = aggregate(datasets["llm_text"])["pass@1"]
    pass_llm_ici = aggregate(datasets["llm_ici"])["pass@1"]

    deltas = {
        "delta_slm": round(pass_slm_ici - pass_slm_text, 3),
        "delta_llm": round(pass_llm_ici - pass_llm_text, 3),
        "gap_before": round(pass_llm_text - pass_slm_text, 3),
        "gap_after": round(pass_llm_ici - pass_slm_ici, 3),
    }
    save_csv(
        os.path.join(args.outdir, "gap_summary.csv"),
        [deltas],
        ["delta_slm", "delta_llm", "gap_before", "gap_after"],
    )

    # Error distributions per bucket
    error_rows = []
    error_maps = {}
    for name, buckets in bucketized.items():
        error_maps[name] = {bucket: error_distribution(rows) for bucket, rows in buckets.items()}
        for bucket in buckets_order:
            dist = error_maps[name][bucket]
            for cat, val in dist.items():
                error_rows.append({
                    "run": name,
                    "bucket": bucket,
                    "category": cat,
                    "proportion": round(val, 3),
                })
    save_csv(os.path.join(args.outdir, "error_breakdown.csv"), error_rows, ["run", "bucket", "category", "proportion"])

    # Plots
    plot_success_vs_ici(buckets_order, aggregated["slm_text"], aggregated["llm_text"], "Pass@1 vs ICI (text-only)", os.path.join(args.figdir, "success_vs_ici_text.png"))
    plot_success_vs_ici(buckets_order, aggregated["slm_ici"], aggregated["llm_ici"], "Pass@1 vs ICI (ICI-aware)", os.path.join(args.figdir, "success_vs_ici_ici.png"))
    plot_gap_curve(buckets_order, gap_before, gap_after, os.path.join(args.figdir, "gap_vs_ici.png"))
    plot_error_stacks(buckets_order, error_maps["slm_text"], error_maps["llm_text"], os.path.join(args.figdir, "error_breakdown_text.png"))
    plot_error_stacks(buckets_order, error_maps["slm_ici"], error_maps["llm_ici"], os.path.join(args.figdir, "error_breakdown_ici.png"))

    print(f"Saved analysis tables to {args.outdir}")
    print(f"Saved figures to {args.figdir}")


if __name__ == "__main__":
    main()
