import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.run_e2e import AgentConfig, ExperimentConfig, build_experiment_config


def test_build_experiment_config_parses_values():
    data = {
        "server_url": "http://localhost:4000",
        "server": {"launch": False},
        "tools_path": "tools.jsonl",
        "tasks": {"path": "tests/fuzzy_tasks.jsonl"},
        "logging": {"out_dir": "runs/test"},
        "model": {"provider": "ollama", "name": "test-model"},
        "retrieval": {"k": 4, "ic_penalty_mu": 0.5, "redundancy_penalty_nu": 0.2},
        "limits": {"max_steps": 3, "max_retries": 1},
        "complexity": {"slot_threshold": 0.7},
    }
    config = build_experiment_config(data)
    assert isinstance(config, ExperimentConfig)
    assert config.agent.max_steps == 3
    assert config.retrieval.k == 4
    assert str(config.tools_path) == "tools.jsonl"
    assert isinstance(config.agent, AgentConfig)
