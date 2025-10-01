#!/usr/bin/env python3
"""Generate aggregated reports from experiment outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _find_summary_files(root: Path) -> List[Path]:
    summaries: List[Path] = []
    for path in root.rglob("summary.json"):
        summaries.append(path)
    bench_summary = root / "bench_summary.json"
    if bench_summary.exists():
        summaries.append(bench_summary)
    return summaries


def _load_summary(path: Path) -> List[Dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return [data]


def build_report(runs_path: Path, out_path: Path) -> Dict[str, Any]:
    summaries = []
    for summary_file in _find_summary_files(runs_path):
        summaries.extend(_load_summary(summary_file))
    if not summaries:
        raise RuntimeError(f"No summary.json found under {runs_path}")
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "metrics.csv"
    fieldnames = sorted({key for item in summaries for key in item.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in summaries:
            writer.writerow(item)
    report_md = out_path / "report.md"
    lines = ["# MCP Agent Benchmark Report", ""]
    lines.append("| Run | Pass@1 | Calls | Avg Latency |")
    lines.append("| --- | --- | --- | --- |")
    for item in summaries:
        name = item.get("output_dir") or item.get("run") or "run"
        pass_at_1 = f"{item.get('pass_at_1_overall', item.get('pass_at_1', 0.0)):.3f}"
        calls = f"{item.get('calls', 0.0):.0f}"
        latency = f"{item.get('avg_latency_ms', 0.0):.1f}"
        lines.append(f"| {name} | {pass_at_1} | {calls} | {latency} |")
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "summaries": len(summaries),
        "csv": str(csv_path),
        "report": str(report_md),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark reports")
    parser.add_argument("--runs", type=Path, required=True, help="Path with run outputs")
    parser.add_argument("--out", type=Path, required=True, help="Destination directory")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_report(args.runs, args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
