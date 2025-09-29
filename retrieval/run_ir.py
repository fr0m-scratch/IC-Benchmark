#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run IR experiment: text-only vs ICI-penalized.
Outputs Top-1/Recall/C@K/nDCG metrics, per-task rankings, lambda sweep,
rescued cases, Top-1 ICI analysis, and candidate lists for downstream eval.
"""
import argparse
import csv
import json
import math
import os
from typing import Dict, List, Sequence, Tuple

from .docs_to_text import tool_to_text_doc
from .tfidf_ir import tfidf_rank


def _read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _metric_order(entry, use_adj: bool) -> List[int]:
    ord_plain, ord_adj, _, _ = entry
    return ord_adj if use_adj else ord_plain


def _metric_scores(entry, use_adj: bool) -> List[float]:
    _, _, scores_plain, scores_adj = entry
    return scores_adj if use_adj else scores_plain


def _eval_metrics(ids: Sequence[str],
                  ranked: Sequence[Tuple[List[int], List[int], List[float], List[float]]],
                  gold_ids: Sequence[str],
                  ks: Sequence[int],
                  use_adj: bool) -> Dict[str, float]:
    totals = {
        "Top-1": 0.0,
        "nDCG@10": 0.0,
    }
    recalls = {k: 0.0 for k in ks}
    completeness = {k: 0.0 for k in ks}

    for entry, gold in zip(ranked, gold_ids):
        order = _metric_order(entry, use_adj)
        try:
            rank = next(i for i, idx in enumerate(order) if ids[idx] == gold)
        except StopIteration:
            rank = None

        if rank is not None:
            if rank == 0:
                totals["Top-1"] += 1.0
            if rank < 10:
                totals["nDCG@10"] += 1.0 / math.log2(rank + 2)
            for k in ks:
                if rank < k:
                    recalls[k] += 1.0
                    completeness[k] += 1.0 / k
        # no contribution otherwise

    n = len(gold_ids) or 1
    metrics = {
        "Top-1": totals["Top-1"] / n,
        "nDCG@10": totals["nDCG@10"] / n,
    }
    for k in ks:
        metrics[f"Recall@{k}"] = recalls[k] / n
        metrics[f"C@{k}"] = completeness[k] / n
    metrics["n"] = n
    return metrics


def _round_metrics(metrics: Dict[str, float], digits: int = 3) -> Dict[str, float]:
    return {k: (round(v, digits) if isinstance(v, float) else v) for k, v in metrics.items()}


def _save_per_task(ids: Sequence[str],
                   ranked_plain,
                   ranked_ici,
                   tasks,
                   ici_map,
                   out_path: str,
                   K: int):
    with open(out_path, "w", encoding="utf-8") as f:
        for plain_entry, ici_entry, task in zip(ranked_plain, ranked_ici, tasks):
            ord_plain, _, scores_plain, _ = plain_entry
            _, ord_ici, _, scores_ici = ici_entry

            top_plain = [ids[idx] for idx in ord_plain[:K]]
            top_ici = [ids[idx] for idx in ord_ici[:K]]

            payload = {
                "task_id": task["task_id"],
                "query": task["query"],
                "gold": task["gold_tool_id"],
                "rank_text": top_plain,
                "rank_ici": top_ici,
                "candidates_text": [
                    {
                        "tool_id": ids[idx],
                        "rank": i + 1,
                        "text_score": scores_plain[idx],
                        "ici": ici_map.get(ids[idx]),
                    }
                    for i, idx in enumerate(ord_plain)
                ],
                "candidates_ici": [
                    {
                        "tool_id": ids[idx],
                        "rank": i + 1,
                        "score": scores_ici[idx],
                        "text_score": scores_plain[idx],
                        "ici": ici_map.get(ids[idx]),
                    }
                    for i, idx in enumerate(ord_ici)
                ],
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _export_candidates(per_task_path: str,
                       out_text: str,
                       out_ici: str,
                       cutoff: int = 5):
    rows = _read_jsonl(per_task_path)
    for out_path, key in ((out_text, "candidates_text"), (out_ici, "candidates_ici")):
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps({
                    "task_id": row["task_id"],
                    "query": row["query"],
                    "gold": row["gold"],
                    "candidates": row[key][:cutoff],
                }, ensure_ascii=False) + "\n")


def _lambda_sweep(queries,
                  id2doc,
                  ici_map,
                  gold,
                  ks,
                  out_csv: str):
    fieldnames = ["lambda", "Top-1", "nDCG@10"]
    for k in ks:
        fieldnames.extend([f"Recall@{k}", f"C@{k}"])

    rows = []
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        ids, ranked = tfidf_rank(queries, id2doc, ici_map, lam)
        use_adj = lam > 0.0
        metrics = _eval_metrics(ids, ranked, gold, ks, use_adj)
        row = {"lambda": lam}
        row.update({k: round(metrics[k], 3) for k in fieldnames if k != "lambda"})
        rows.append(row)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rescued(ids, ranked_plain, ranked_ici, tasks, K, out_jsonl: str):
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for plain_entry, ici_entry, task in zip(ranked_plain, ranked_ici, tasks):
            ord_plain, _, _, _ = plain_entry
            _, ord_ici, _, _ = ici_entry
            top_plain = [ids[idx] for idx in ord_plain[:K]]
            top_ici = [ids[idx] for idx in ord_ici[:K]]
            if task["gold_tool_id"] not in top_plain and task["gold_tool_id"] in top_ici:
                f.write(json.dumps({
                    "task_id": task["task_id"],
                    "query": task["query"],
                    "gold": task["gold_tool_id"],
                    "topK_plain": top_plain,
                    "topK_ici": top_ici,
                }, ensure_ascii=False) + "\n")


def _mean(vals: List[float]) -> float:
    return round(sum(vals) / len(vals), 3) if vals else float("nan")


def _top1_ici_analysis(ids,
                       ranked_plain,
                       ranked_ici,
                       gold,
                       ici_map,
                       lam: float,
                       out_csv: str):
    rows = []
    for label, ranked, use_adj, lam_value in (
        ("text", ranked_plain, False, 0.0),
        ("ici", ranked_ici, True, lam),
    ):
        top_ids = [ids[_metric_order(entry, use_adj)[0]] for entry in ranked]
        icis = [ici_map[i] for i in top_ids]
        right = [ici_map[i] for i, g in zip(top_ids, gold) if i == g]
        wrong = [ici_map[i] for i, g in zip(top_ids, gold) if i != g]
        acc = sum(1 for i, g in zip(top_ids, gold) if i == g) / len(gold)
        rows.append({
            "lambda": lam_value,
            "mode": label,
            "Top1_acc": round(acc, 3),
            "Top1_mean_ICI": _mean(icis),
            "Top1_right_ICI": _mean(right),
            "Top1_wrong_ICI": _mean(wrong),
        })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["lambda", "mode", "Top1_acc", "Top1_mean_ICI", "Top1_right_ICI", "Top1_wrong_ICI"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Run TF-IDF vs ICI-penalized retrieval comparison.")
    parser.add_argument("--tools", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--ici", required=True, help="JSONL features with precomputed ICI scores")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.5)
    parser.add_argument("--K", type=int, default=3, help="Top-K cutoff for primary metrics")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--candidate-cutoff", type=int, default=5, help="How many candidates to export per task")
    args = parser.parse_args()

    _ensure_dir(args.outdir)

    tools = _read_jsonl(args.tools)
    tasks = _read_jsonl(args.tasks)
    feats = _read_jsonl(args.ici)
    ici_map = {row["tool_id"]: row["ici"] for row in feats}

    id2doc = {tool["tool_id"]: tool_to_text_doc(tool) for tool in tools}
    queries = [task["query"] for task in tasks]
    gold = [task["gold_tool_id"] for task in tasks]

    ids, ranked_plain = tfidf_rank(queries, id2doc, ici_map=None, lam=0.0)
    _, ranked_ici = tfidf_rank(queries, id2doc, ici_map=ici_map, lam=args.lam)

    eval_ks = sorted({3, 5, args.K})

    metrics_plain = _eval_metrics(ids, ranked_plain, gold, eval_ks, use_adj=False)
    metrics_ici = _eval_metrics(ids, ranked_ici, gold, eval_ks, use_adj=True)

    summary_path = os.path.join(args.outdir, "preliminary_ir_results.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["Metric", "Text-only", f"ICI-penalized (λ={args.lam})"]
        writer.writerow(header)
        for key in [
            "Top-1",
            "Recall@3",
            "Recall@5",
            "C@3",
            "C@5",
            "nDCG@10",
        ]:
            writer.writerow([key, round(metrics_plain.get(key, 0.0), 3), round(metrics_ici.get(key, 0.0), 3)])

    per_task_path = os.path.join(args.outdir, "per_task_rankings.jsonl")
    _save_per_task(ids, ranked_plain, ranked_ici, tasks, ici_map, per_task_path, args.K)

    _export_candidates(
        per_task_path,
        out_text=os.path.join(args.outdir, "candidates_text.jsonl"),
        out_ici=os.path.join(args.outdir, "candidates_ici.jsonl"),
        cutoff=args.candidate_cutoff,
    )

    _lambda_sweep(
        queries,
        id2doc,
        ici_map,
        gold,
        eval_ks,
        os.path.join(args.outdir, "lambda_sweep.csv"),
    )
    _rescued(
        ids,
        ranked_plain,
        ranked_ici,
        tasks,
        args.K,
        os.path.join(args.outdir, "rescued_cases.jsonl"),
    )
    _top1_ici_analysis(
        ids,
        ranked_plain,
        ranked_ici,
        gold,
        ici_map,
        args.lam,
        os.path.join(args.outdir, "top1_ici_analysis.csv"),
    )

    print(f"Done. Results -> {args.outdir}")
    print(
        "Summary:" +
        f" Top-1 text-only={metrics_plain['Top-1']:.3f}, ICI={metrics_ici['Top-1']:.3f};"+
        f" Recall@3 text-only={metrics_plain['Recall@3']:.3f}, ICI={metrics_ici['Recall@3']:.3f};"+
        f" Recall@5 text-only={metrics_plain['Recall@5']:.3f}, ICI={metrics_ici['Recall@5']:.3f}"
    )


if __name__ == "__main__":
    main()
