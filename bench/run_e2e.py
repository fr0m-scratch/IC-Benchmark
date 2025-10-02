#!/usr/bin/env python3
"""End-to-end runner for MCP agent experiments."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

from analysis.ic_score import ToolICResult, compute_all_tool_scores, load_tools
from bench.agent_executor import AgentConfig, AgentExecutor, ToolInvoker
from bench.metrics import MetricsTracker
from bench.providers import ProviderConfig, build_provider
from bench.retrieval import RetrievalConfig, ToolRetriever

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


@dataclass
class ExperimentConfig:
    server_url: str
    server_launch: bool
    server_cmd: List[str]
    tools_path: Path
    tasks_path: Path
    output_dir: Path
    provider: ProviderConfig
    agent: AgentConfig
    retrieval: RetrievalConfig


def load_config(path: Path) -> ExperimentConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    data = _parse_config_text(text, path)
    return build_experiment_config(data)


def build_experiment_config(data: Dict[str, Any]) -> ExperimentConfig:
    server = data.get("server", {})
    launch = server.get("launch", False)
    server_cmd = server.get("cmd")
    if launch and not server_cmd:
        base_cmd = [sys.executable, str((Path(__file__).resolve().parents[1] / "server" / "app.py"))]
        if "port" in server:
            base_cmd.extend(["--port", str(server["port"])])
        if "tools_path" in server:
            base_cmd.extend(["--tools", str(server["tools_path"])])
        if server.get("wrap"):
            base_cmd.extend(["--wrap", server["wrap"]])
        if server.get("default_latency"):
            base_cmd.extend(["--default-latency", str(server["default_latency"])])
        if server.get("validate_output"):
            base_cmd.append("--validate-output")
        if server.get("require_auth"):
            base_cmd.append("--require-auth")
            if server.get("auth_key"):
                base_cmd.extend(["--auth-key", str(server["auth_key"])])
        if server.get("rate_limit"):
            base_cmd.extend(["--rate-limit", str(server["rate_limit"])])
        if server.get("paginate"):
            base_cmd.append("--paginate")
            if server.get("page_size"):
                base_cmd.extend(["--page-size", str(server["page_size"])])
        server_cmd = base_cmd
    server_url = data.get("server_url") or f"http://localhost:{server.get('port', 3001)}"
    tools_path = Path(data.get("tools_path") or server.get("tools_path") or data.get("server", {}).get("tools"))
    if not tools_path:
        raise ValueError("tools_path must be defined in config")
    tasks_path = Path(data.get("tasks", {}).get("path"))
    if not tasks_path:
        raise ValueError("tasks.path must be provided")
    out_dir = Path(data.get("logging", {}).get("out_dir", "runs/output"))
    provider_cfg = ProviderConfig(
        provider=data.get("model", {}).get("provider", "ollama"),
        model=data.get("model", {}).get("name", "qwen2.5:7b-instruct"),
        base_url=data.get("model", {}).get("base_url"),
        api_key=data.get("model", {}).get("api_key"),
        timeout=int(data.get("model", {}).get("timeout", 120)),
    )
    retrieval_cfg = RetrievalConfig(
        k=int(data.get("retrieval", {}).get("k", 6)),
        ic_penalty_mu=float(data.get("retrieval", {}).get("ic_penalty_mu", 0.6)),
        redundancy_penalty_nu=float(data.get("retrieval", {}).get("redundancy_penalty_nu", 0.3)),
    )
    limits = data.get("limits", {})
    complexity = data.get("complexity", {})
    agent_cfg = AgentConfig(
        max_steps=int(limits.get("max_steps", 6)),
        max_retries=int(limits.get("max_retries", 2)),
        retrieval_k=retrieval_cfg.k,
        ic_penalty_mu=retrieval_cfg.ic_penalty_mu,
        redundancy_penalty_nu=retrieval_cfg.redundancy_penalty_nu,
        slot_threshold=float(complexity.get("slot_threshold", 0.65)),
        gate_threshold=float(complexity.get("gate_threshold", 0.05)),
        temperature=float(data.get("model", {}).get("temperature", 0.2)),
    )
    return ExperimentConfig(
        server_url=server_url,
        server_launch=launch,
        server_cmd=server_cmd or [],
        tools_path=tools_path,
        tasks_path=tasks_path,
        output_dir=out_dir,
        provider=provider_cfg,
        agent=agent_cfg,
        retrieval=retrieval_cfg,
    )


def _parse_config_text(text: str, path: Path) -> Dict[str, Any]:
    if yaml:
        return yaml.safe_load(text) or {}
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse config {path}. Install PyYAML (pip install pyyaml) or provide JSON."
        ) from exc


def _load_tasks(path: Path) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
    return tasks


def _wait_for_server(url: str, timeout: float = 15.0) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{url.rstrip('/')}/health", timeout=1.0)
            if resp.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Server did not become healthy at {url} within {timeout}s")


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    server_proc: Optional[subprocess.Popen] = None
    try:
        if config.server_launch and config.server_cmd:
            server_proc = subprocess.Popen(config.server_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _wait_for_server(config.server_url)
        tools = load_tools(config.tools_path)
        tool_results = compute_all_tool_scores(tools)
        ic_map: Dict[str, ToolICResult] = {result.tool.tool_id: result for result in tool_results}
        retriever = ToolRetriever(config.retrieval)
        retriever.build(tools, ic_map)
        provider = build_provider(config.provider)
        invoker = ToolInvoker(config.server_url)
        agent = AgentExecutor(config.agent, provider, retriever, ic_map, invoker)
        tasks = _load_tasks(config.tasks_path)
        results_path = config.output_dir / "results.jsonl"
        metrics_tracker = MetricsTracker()
        successes = 0.0
        total_calls = 0.0
        selections_total = 0.0
        selections_correct = 0.0
        with results_path.open("w", encoding="utf-8") as sink:
            for task in tasks:
                prompt = task.get("instruction") or task.get("prompt") or task.get("query")
                if not prompt:
                    continue
                result = agent.run(prompt)
                sink.write(json.dumps({
                    "task": task.get("id"),
                    "prompt": prompt,
                    "final_answer": result.get("final_answer"),
                    "telemetry": result.get("telemetry"),
                    "metrics": result.get("metrics"),
                }, ensure_ascii=False) + "\n")
                summary = result.get("metrics", {})
                calls = summary.get("calls", 0.0)
                total_calls += calls
                successes += summary.get("pass_at_1", 0.0) * calls
                metrics_tracker.logs.extend(agent.metrics.logs)
                metrics_tracker.add_tokens(summary.get("token_input", 0.0), summary.get("token_output", 0.0))
                agent.metrics.logs.clear()
                # Tool selection accuracy (if ground-truth available)
                expected = task.get("expect") or {}
                expected_tool = None
                if isinstance(expected, dict):
                    if expected.get("tool"):
                        expected_tool = expected.get("tool")
                    else:
                        seq = expected.get("tool_sequence") or []
                        if isinstance(seq, list) and seq:
                            expected_tool = seq[-1]
                selected_tool = None
                telemetry: List[Dict[str, Any]] = result.get("telemetry") or []  # type: ignore[name-defined]
                for event in telemetry:
                    if event.get("event") == "gate":
                        selected_tool = event.get("selected_tool")
                        break
                if expected_tool and selected_tool:
                    selections_total += 1.0
                    if expected_tool == selected_tool:
                        selections_correct += 1.0
        aggregated = metrics_tracker.summarise()
        aggregated.update({
            "tasks": len(tasks),
            "pass_at_1_overall": successes / total_calls if total_calls else 0.0,
            "selections": selections_total,
            "tool_select_acc": selections_correct / selections_total if selections_total else 0.0,
        })
        summary_path = config.output_dir / "summary.json"
        summary_path.write_text(json.dumps(aggregated, indent=2), encoding="utf-8")
        return aggregated
    finally:
        if server_proc:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end MCP agent evaluation")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    summary = run_experiment(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
