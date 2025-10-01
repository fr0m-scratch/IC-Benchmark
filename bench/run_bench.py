#!/usr/bin/env python3
"""Grid runner for MCP agent experiments."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from bench import run_e2e

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


def _parse_file(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml:
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-") or "run"


def run_grid(config_path: Path) -> List[Dict[str, Any]]:
    data = _parse_file(config_path)
    grid = data.pop("grid", {})
    models = grid.get("models") or []
    conditions = grid.get("conditions") or []
    if not models or not conditions:
        raise ValueError("grid.models and grid.conditions must be defined")

    base_logging = Path(data.get("logging", {}).get("out_dir", "runs/bench"))
    base_retrieval = data.get("retrieval", {}).copy()
    summaries: List[Dict[str, Any]] = []
    for model in models:
        for condition in conditions:
            combo = copy.deepcopy(data)
            combo.setdefault("model", {}).update(model)
            combo.setdefault("retrieval", {}).update(base_retrieval)
            combo.setdefault("logging", {})
            model_slug = _slug(model.get("name", model.get("provider", "model")))
            cond_slug = _slug(condition.get("name", "cond"))
            combo["logging"]["out_dir"] = str(base_logging / f"{model_slug}_{cond_slug}")
            combo.setdefault("server", {}).update({k: v for k, v in condition.items() if k in {"wrap", "rate_limit", "paginate"}})
            if "retrieval_ic" in condition:
                combo.setdefault("retrieval", {})
                combo["retrieval"]["ic_penalty_mu"] = base_retrieval.get("ic_penalty_mu", 0.6) if condition["retrieval_ic"] else 0.0
            if "slot_policy" in condition:
                slot_mode = condition["slot_policy"]
                combo.setdefault("complexity", {})
                if slot_mode == "slot_by_slot":
                    combo["complexity"]["slot_threshold"] = 0.0
                else:
                    combo["complexity"]["slot_threshold"] = 1.0
            if "gate_threshold" in condition:
                combo.setdefault("complexity", {})
                combo["complexity"]["gate_threshold"] = condition["gate_threshold"]
            experiment_config = run_e2e.build_experiment_config(combo)
            summary = run_e2e.run_experiment(experiment_config)
            summary.update({
                "model": model,
                "condition": condition.get("name", "condition"),
                "output_dir": combo["logging"]["out_dir"],
            })
            summaries.append(summary)
    output_path = base_logging / "bench_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return summaries


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark grid")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    summaries = run_grid(args.config)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
