"""Complexity-aware agent executor."""
from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import requests

from analysis.ic_score import ToolICResult, ToolRecord
from bench.arguments import (
    find_missing_required,
    full_argument_prompt,
    should_use_slot_mode,
    slot_fill_prompt,
    synthesise_minimal_arguments,
    validate_arguments,
)
from bench.metrics import InvocationLog, MetricsTracker
from bench.providers import ChatMessage, ChatProvider, _extract_json
from bench.retrieval import RetrievedTool, ToolRetriever

PROMPTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "prompts"


@dataclass
class AgentConfig:
    max_steps: int = 6
    max_retries: int = 2
    retrieval_k: int = 8
    ic_penalty_mu: float = 0.6
    redundancy_penalty_nu: float = 0.3
    slot_threshold: float = 0.65
    gate_threshold: float = 0.05
    temperature: float = 0.2


@dataclass
class AgentTelemetry:
    steps: List[Dict[str, Any]] = field(default_factory=list)

    def log(self, event: str, payload: Dict[str, Any]) -> None:
        self.steps.append({"event": event, **payload})


@dataclass
class InvocationResult:
    success: bool
    output: Dict[str, Any]
    error: Optional[Dict[str, Any]]
    status: str
    latency_ms: float


class ToolInvoker:
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")
        self.session = requests.Session()

    def invoke(self, tool_name: str, arguments: Dict[str, Any]) -> InvocationResult:
        url = f"{self.server_url}/invoke"
        payload = {"tool": tool_name, "input": arguments}
        start = time.perf_counter()
        try:
            resp = self.session.post(url, json=payload, timeout=60)
        except requests.RequestException as exc:
            return InvocationResult(False, {}, {"code": "network_error", "message": str(exc)}, "network_error", 0.0)
        latency_ms = (time.perf_counter() - start) * 1000
        try:
            data = resp.json()
        except Exception:
            data = {"ok": False, "error": {"code": "invalid_response", "message": resp.text[:200]}}
        if resp.status_code >= 400 or not data.get("ok"):
            error = data.get("error") or {"code": "unknown", "message": resp.text[:200]}
            return InvocationResult(False, {}, error, "server_error", latency_ms)
        return InvocationResult(True, data.get("output", {}), None, "ok", latency_ms)


class AgentExecutor:
    def __init__(
        self,
        config: AgentConfig,
        provider: ChatProvider,
        retriever: ToolRetriever,
        ic_results: Dict[str, ToolICResult],
        invoker: ToolInvoker,
    ):
        self.config = config
        self.provider = provider
        self.retriever = retriever
        self.ic_results = ic_results
        self.invoker = invoker
        self.telemetry = AgentTelemetry()
        self.metrics = MetricsTracker()
        self.system_prompt = _load_prompt("agent_system.txt")
        self.slot_prompt = _load_prompt("slot_filling.txt")
        self.validator_prompt = _load_prompt("validator_feedback.txt")

    def run(self, prompt: str) -> Dict[str, Any]:
        self.telemetry.steps.clear()
        self.metrics.logs.clear()
        context = prompt
        scratchpad = []
        for step in range(self.config.max_steps):
            self.telemetry.log("plan", {"step": step, "context": context[-500:]})
            candidates = self.retriever.query(context, k=self.config.retrieval_k)
            self.telemetry.log(
                "retrieve",
                {
                    "step": step,
                    "candidates": [
                        {
                            "tool_id": c.tool.tool_id,
                            "score": c.score,
                            "similarity": c.similarity,
                            "ic": c.ic_score,
                        }
                        for c in candidates
                    ],
                },
            )
            if not candidates:
                break
            selection = self._select_tool(candidates)
            use_tool = self._gate(selection)
            self.telemetry.log(
                "gate",
                {"step": step, "selected_tool": selection.tool.tool_id, "decision": use_tool, "score": selection.score},
            )
            if not use_tool:
                break
            arguments, schema_valid = self._fill_arguments(selection, prompt, context)
            self.telemetry.log(
                "arguments",
                {
                    "step": step,
                    "tool_id": selection.tool.tool_id,
                    "arguments": arguments,
                    "schema_valid": schema_valid,
                },
            )
            retries = 0
            while retries <= self.config.max_retries:
                result = self.invoker.invoke(selection.tool.name, arguments)
                self.telemetry.log(
                    "invoke",
                    {
                        "step": step,
                        "tool_id": selection.tool.tool_id,
                        "success": result.success,
                        "error": result.error,
                        "latency_ms": result.latency_ms,
                    },
                )
                self.metrics.record(
                    InvocationLog(
                        tool_id=selection.tool.tool_id,
                        success=result.success,
                        schema_valid=schema_valid,
                        retries=retries,
                        ic_score=selection.ic_score,
                        latency_ms=result.latency_ms,
                    )
                )
                if result.success:
                    scratchpad.append(
                        {
                            "tool_id": selection.tool.tool_id,
                            "arguments": arguments,
                            "output": result.output,
                        }
                    )
                    summary = json.dumps(result.output, ensure_ascii=False)
                    context += f"\nTool {selection.tool.tool_id} -> {summary}"
                    break
                retries += 1
                if retries > self.config.max_retries:
                    break
                arguments, schema_valid = self._repair_arguments(selection, arguments, result.error or {})
            if retries > self.config.max_retries:
                break
        final_answer = self._finalize(prompt, scratchpad)
        self.telemetry.log("finalize", {"answer": final_answer})
        return {
            "final_answer": final_answer,
            "telemetry": self.telemetry.steps,
            "metrics": self.metrics.summarise(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_tool(self, candidates: Sequence[RetrievedTool]) -> RetrievedTool:
        return max(candidates, key=lambda c: c.score)

    def _gate(self, candidate: RetrievedTool) -> bool:
        return candidate.score >= self.config.gate_threshold or candidate.similarity > 0.0

    def _fill_arguments(
        self,
        candidate: RetrievedTool,
        task_prompt: str,
        context: str,
    ) -> (Dict[str, Any], bool):
        schema = candidate.tool.input_schema or {}
        task_context = f"Task: {task_prompt}\nContext: {context[-500:]}"
        messages = [
            ChatMessage("system", self.system_prompt),
            ChatMessage("user", full_argument_prompt(schema, task_context)),
        ]
        response = self.provider.generate(messages, temperature=self.config.temperature)
        arguments = _extract_json(response.text)
        if not isinstance(arguments, dict):
            arguments = {}
        valid, _ = validate_arguments(schema, arguments)
        if valid:
            return arguments, True
        if should_use_slot_mode(candidate.ic_score, self.config.slot_threshold):
            return self._slot_fill(schema, task_context, arguments)
        if not arguments:
            arguments = synthesise_minimal_arguments(schema)
        return arguments, False

    def _slot_fill(
        self,
        schema: Dict[str, Any],
        context: str,
        arguments: Dict[str, Any],
    ) -> (Dict[str, Any], bool):
        attempts = 0
        current = dict(arguments)
        while attempts < 5:
            missing = find_missing_required(schema, current)
            if not missing:
                break
            prompt = self.slot_prompt + "\n\n" + slot_fill_prompt(schema, current)
            messages = [
                ChatMessage("system", self.system_prompt),
                ChatMessage("user", prompt + f"\nContext: {context}"),
            ]
            response = self.provider.generate(messages, temperature=0.0)
            update = _extract_json(response.text)
            if not isinstance(update, dict) or "slot" not in update:
                break
            slot = update["slot"].lstrip("/")
            value = update.get("value")
            current[slot] = value
            valid, _ = validate_arguments(schema, current)
            if valid:
                return current, True
            attempts += 1
        return current, False

    def _repair_arguments(
        self,
        candidate: RetrievedTool,
        arguments: Dict[str, Any],
        error: Any,
    ) -> (Dict[str, Any], bool):
        schema = candidate.tool.input_schema or {}
        if isinstance(error, dict):
            code = error.get("code", "unknown")
            message = error.get("message", "")
        elif isinstance(error, str):
            code = "unknown"
            message = error
        else:
            code = "unknown"
            message = str(error)
        summary = f"Error: {code} - {message}"
        prompt = self.validator_prompt.replace("<schema_violation_summary>", summary)
        messages = [
            ChatMessage("system", self.system_prompt),
            ChatMessage("user", prompt),
        ]
        response = self.provider.generate(messages, temperature=0.0)
        fix = _extract_json(response.text)
        if not isinstance(fix, dict):
            return arguments, False
        if "slot" in fix and "value" in fix:
            updated = dict(arguments)
            updated[fix["slot"].lstrip("/")] = fix["value"]
            valid, _ = validate_arguments(schema, updated)
            return updated, valid
        valid, _ = validate_arguments(schema, fix)
        return (fix if valid else arguments), valid

    def _finalize(self, prompt: str, scratchpad: List[Dict[str, Any]]) -> str:
        if not scratchpad:
            return f"No tool call executed. Prompt: {prompt}"
        parts = [f"Tool {item['tool_id']} output: {json.dumps(item['output'], ensure_ascii=False)}" for item in scratchpad]
        return " \n".join(parts)


def _load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


__all__ = ["AgentExecutor", "AgentConfig", "ToolInvoker"]
