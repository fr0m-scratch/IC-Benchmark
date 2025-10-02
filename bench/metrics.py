"""Metric tracking utilities for agent runs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class InvocationLog:
    tool_id: str
    success: bool
    schema_valid: bool
    retries: int
    ic_score: float
    latency_ms: float


@dataclass
class MetricsTracker:
    logs: List[InvocationLog] = field(default_factory=list)
    token_input: float = 0.0
    token_output: float = 0.0

    def record(self, log: InvocationLog) -> None:
        self.logs.append(log)

    def add_tokens(self, prompt_tokens: float, completion_tokens: float) -> None:
        self.token_input += float(prompt_tokens or 0.0)
        self.token_output += float(completion_tokens or 0.0)

    def _bucket(self, ic_score: float) -> str:
        if ic_score < 4.0:
            return "low"
        if ic_score < 9.0:
            return "medium"
        return "high"

    def summarise(self) -> Dict[str, float]:
        if not self.logs:
            return {
                "calls": 0,
                "steps": 0,
                "pass_at_1": 0.0,
                "first_try_valid": 0.0,
                "avg_retries": 0.0,
                "avg_latency_ms": 0.0,
                "tool_switches": 0,
                "token_input": 0.0,
                "token_output": 0.0,
            }
        calls = len(self.logs)
        successes = sum(1 for log in self.logs if log.success)
        schema_valid = sum(1 for log in self.logs if log.schema_valid)
        retries = sum(log.retries for log in self.logs)
        latency = sum(log.latency_ms for log in self.logs)
        switches = 0
        last_tool = None
        for log in self.logs:
            if last_tool is not None and log.tool_id != last_tool:
                switches += 1
            last_tool = log.tool_id
        summary = {
            "calls": float(calls),
            "steps": float(calls),
            "pass_at_1": successes / calls,
            "first_try_valid": schema_valid / calls,
            "avg_retries": retries / calls,
            "avg_latency_ms": latency / calls,
            "tool_switches": float(switches),
            "token_input": float(self.token_input),
            "token_output": float(self.token_output),
        }
        # bucketed success rates
        bucketed: Dict[str, List[InvocationLog]] = {}
        for log in self.logs:
            bucket = self._bucket(log.ic_score)
            bucketed.setdefault(bucket, []).append(log)
        for bucket, entries in bucketed.items():
            if not entries:
                continue
            bucket_success = sum(1 for entry in entries if entry.success) / len(entries)
            summary[f"pass_at_1_{bucket}"] = bucket_success
        return summary


__all__ = ["InvocationLog", "MetricsTracker"]
