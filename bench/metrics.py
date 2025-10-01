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

    def record(self, log: InvocationLog) -> None:
        self.logs.append(log)

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
                "pass_at_1": 0.0,
                "first_try_valid": 0.0,
                "avg_retries": 0.0,
                "avg_latency_ms": 0.0,
            }
        calls = len(self.logs)
        successes = sum(1 for log in self.logs if log.success)
        schema_valid = sum(1 for log in self.logs if log.schema_valid)
        retries = sum(log.retries for log in self.logs)
        latency = sum(log.latency_ms for log in self.logs)
        summary = {
            "calls": float(calls),
            "pass_at_1": successes / calls,
            "first_try_valid": schema_valid / calls,
            "avg_retries": retries / calls,
            "avg_latency_ms": latency / calls,
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
