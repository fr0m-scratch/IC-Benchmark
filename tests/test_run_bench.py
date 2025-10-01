import json
import pathlib
import sys
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import run_bench


def test_run_grid_invokes_experiments(tmp_path):
    config_path = tmp_path / "bench.yaml"
    config = {
        "server": {"launch": False, "tools_path": "tools.jsonl"},
        "retrieval": {"k": 2, "ic_penalty_mu": 0.6, "redundancy_penalty_nu": 0.3},
        "limits": {"max_steps": 2, "max_retries": 1},
        "tasks": {"path": "tests/fuzzy_tasks.jsonl"},
        "logging": {"out_dir": "runs/test"},
        "model": {"provider": "ollama", "name": "stub"},
        "grid": {
            "models": [{"provider": "ollama", "name": "stub"}],
            "conditions": [{"name": "BASE", "retrieval_ic": False, "wrap": "none", "slot_policy": "simple"}],
        },
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    summaries = []

    def fake_run_experiment(config):
        summaries.append(config.output_dir)
        return {"calls": 0.0, "pass_at_1_overall": 0.0, "avg_latency_ms": 0.0}

    with mock.patch("bench.run_e2e.run_experiment", side_effect=fake_run_experiment):
        result = run_bench.run_grid(config_path)

    assert summaries, "Expected run_experiment to be invoked"
    assert result[0]["output_dir"].endswith("stub_base")
