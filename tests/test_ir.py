import json
import os
import subprocess
import sys


def test_run_ir_smoke():
    tools = "data/prelim/tools.jsonl"
    tasks = "data/prelim/tasks.jsonl"
    feat = "data/prelim/tools.features_ici.jsonl"
    outd = "data/prelim/results_test"

    os.makedirs(outd, exist_ok=True)
    subprocess.check_call([sys.executable, "ici/extract.py", "--tools", tools, "--out", feat + ".tmp"])
    subprocess.check_call([sys.executable, "ici/score.py", "--feat", feat + ".tmp", "--out", feat])
    os.remove(feat + ".tmp")

    subprocess.check_call([
        sys.executable,
        "-m",
        "retrieval.run_ir",
        "--tools",
        tools,
        "--tasks",
        tasks,
        "--ici",
        feat,
        "--lambda",
        "0.5",
        "--K",
        "3",
        "--outdir",
        outd,
    ])

    summary_path = os.path.join(outd, "preliminary_ir_results.csv")
    sweep_path = os.path.join(outd, "lambda_sweep.csv")
    cand_text = os.path.join(outd, "candidates_text.jsonl")
    cand_ici = os.path.join(outd, "candidates_ici.jsonl")
    per_task = os.path.join(outd, "per_task_rankings.jsonl")

    assert os.path.exists(summary_path)
    assert os.path.exists(sweep_path)
    assert os.path.exists(cand_text)
    assert os.path.exists(cand_ici)

    header = open(sweep_path, "r", encoding="utf-8").readline().strip().split(",")
    assert "C@3" in header
    assert "nDCG@10" in header

    sample = json.loads(next(line for line in open(per_task, "r", encoding="utf-8") if line.strip()))
    assert "candidates_text" in sample and isinstance(sample["candidates_text"], list)

    summary_rows = [line.strip().split(",") for line in open(summary_path, "r", encoding="utf-8") if line.strip()]
    metric_names = {row[0] for row in summary_rows[1:]}
    assert {"Top-1", "Recall@3", "C@3", "nDCG@10"}.issubset(metric_names)
