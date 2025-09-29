import json
import os
import subprocess
import sys
import tempfile

def ensure_candidates():
    tools = "data/prelim/tools.jsonl"
    tasks = "data/prelim/tasks.jsonl"
    feat = "data/prelim/tools.features_ici.jsonl"
    outd = "data/prelim/results"
    if not os.path.exists(os.path.join(outd, "candidates_text.jsonl")):
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


def test_e2e_eval_mock():
    ensure_candidates()
    candidates_path = "data/prelim/results/candidates_text.jsonl"

    tasks = [json.loads(line) for line in open("data/prelim/tasks.jsonl", "r", encoding="utf-8") if line.strip()][:2]

    mock_fd, mock_path = tempfile.mkstemp(suffix=".jsonl")
    os.close(mock_fd)
    with open(mock_path, "w", encoding="utf-8") as f:
        for task in tasks:
            payload = {
                "task_id": task["task_id"],
                "response": json.dumps({"tool_id": task["gold_tool_id"], "arguments": {}}),
            }
            f.write(json.dumps(payload) + "\n")

    outdir = tempfile.mkdtemp(prefix="e2e_eval_")
    subprocess.check_call([
        sys.executable,
        "agent/e2e_eval.py",
        "--tasks",
        "data/prelim/tasks.jsonl",
        "--tools",
        "data/prelim/tools.jsonl",
        "--candidates",
        candidates_path,
        "--provider",
        "ollama",
        "--model",
        "mock-model",
        "--mock-responses",
        mock_path,
        "--limit",
        "2",
        "--outdir",
        outdir,
        "--tag",
        "test",
    ])

    per_task = os.path.join(outdir, "per_task_test.jsonl")
    summary = os.path.join(outdir, "summary_test.csv")
    assert os.path.exists(per_task)
    assert os.path.exists(summary)

    summary_lines = [line.strip().split(",") for line in open(summary, "r", encoding="utf-8") if line.strip()]
    metrics = {row[0]: row[1] for row in summary_lines[1:]}
    assert float(metrics["pass@1"]) >= 0.0

    os.remove(mock_path)
