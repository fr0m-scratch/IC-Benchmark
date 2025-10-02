#!/usr/bin/env python3
"""Single-prompt CLI wrapper for the complexity-aware agent.

This mirrors the historical interface used in docs, so you can quickly run a
one-off prompt against a running MCP-style HTTP server.

Example:

PYTHONPATH=. python3 bench/agent.py \
  --server http://localhost:3001 \
  --provider ollama --model qwen2.5:7b-instruct --ollama http://localhost:11434 \
  --prompt "Add 2 and 3." --max-steps 3 --retry 1
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Iterable, List, Optional

import requests

from analysis.ic_score import ToolICResult, ToolRecord, score_tool
from bench.agent_executor import AgentConfig, AgentExecutor, ToolInvoker
from bench.providers import ProviderConfig, build_provider
from bench.retrieval import RetrievalConfig, ToolRetriever


def _fetch_all_tools(server_url: str) -> List[Dict[str, Any]]:
    base = server_url.rstrip("/")
    # Try non-paginated first
    try:
        resp = requests.get(f"{base}/tools", timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "tools" in data:
            return list(data.get("tools") or [])
    except Exception:
        pass
    # Paginated fallback
    tools: List[Dict[str, Any]] = []
    cursor: Optional[int] = 0
    while True:
        params = {"cursor": cursor or 0, "limit": 100}
        resp = requests.get(f"{base}/tools", params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        page = list(data.get("tools") or [])
        tools.extend(page)
        cursor = data.get("next_cursor")
        if not cursor:
            break
    return tools


def _to_tool_records(payloads: Iterable[Dict[str, Any]]) -> List[ToolRecord]:
    records: List[ToolRecord] = []
    for raw in payloads:
        input_schema = raw.get("input_schema") or raw.get("inputSchema")
        if not isinstance(input_schema, dict):
            continue
        name = str(raw.get("name") or raw.get("tool_name") or raw.get("id") or "").strip()
        if not name:
            continue
        server_id = str(raw.get("server_id") or raw.get("serverId") or raw.get("server") or "server")
        tool_id = str(raw.get("tool_id") or f"{server_id}::{name}")
        record = ToolRecord(
            tool_id=tool_id,
            name=name,
            description=str(raw.get("description") or raw.get("tool_description") or ""),
            input_schema=input_schema,
            output_schema=raw.get("output_schema") if isinstance(raw.get("output_schema"), dict) else None,
            server_id=server_id,
            server_name=str(raw.get("server_name") or raw.get("serverName") or server_id),
            raw=raw,
        )
        records.append(record)
    return records


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single prompt with the MCP agent")
    parser.add_argument("--server", required=True, help="Server base URL, e.g. http://localhost:3001")
    parser.add_argument("--prompt", required=True, help="User prompt to solve")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "gemini", "fireworks"]) \
          
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    parser.add_argument("--ollama", dest="ollama_base", help="Ollama base URL (if provider=ollama)")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--retry", dest="max_retries", type=int, default=1)
    parser.add_argument("--k", dest="retrieval_k", type=int, default=6)
    parser.add_argument("--mu", dest="ic_penalty_mu", type=float, default=0.6)
    parser.add_argument("--nu", dest="redundancy_penalty_nu", type=float, default=0.3)
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)

    # Provider
    provider_cfg = ProviderConfig(
        provider=args.provider,
        model=args.model,
        base_url=args.ollama_base,
    )
    provider = build_provider(provider_cfg)

    # Catalogue -> retriever
    payloads = _fetch_all_tools(args.server)
    tools = _to_tool_records(payloads)
    ic_map: Dict[str, ToolICResult] = {t.tool_id: score_tool(t) for t in tools}
    retriever = ToolRetriever(
        RetrievalConfig(k=args.retrieval_k, ic_penalty_mu=args.ic_penalty_mu, redundancy_penalty_nu=args.redundancy_penalty_nu)
    )
    retriever.build(tools, ic_map)

    # Agent
    agent = AgentExecutor(
        AgentConfig(
            max_steps=args.max_steps,
            max_retries=args.max_retries,
            retrieval_k=args.retrieval_k,
            ic_penalty_mu=args.ic_penalty_mu,
            redundancy_penalty_nu=args.redundancy_penalty_nu,
            temperature=args.temperature,
        ),
        provider,
        retriever,
        ic_map,
        ToolInvoker(args.server),
    )

    result = agent.run(args.prompt)
    print(json.dumps({
        "final_answer": result.get("final_answer"),
        "metrics": result.get("metrics"),
        "telemetry": result.get("telemetry"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

