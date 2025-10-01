# Architecture Overview

## Components
- **IC Scorer (`analysis/`)**: loads tool catalogues, extracts schema features, and emits tool/server/set complexity scores with JSON contracts.
- **Task Generator (`bench/taskgen.py`)**: mines dependencies, samples schema-conforming arguments, and emits both generic and fuzzy task suites.
- **Agent Core (`bench/agent_executor.py`)**: executes the plan → retrieve → gate → select → fill → invoke → repair loop with complexity-aware thresholds and rich telemetry.
- **Server (`server_py/app.py`)**: HTTP MCP façade with schema validation, IC-driven wrappers, auth/pagination/rate-limit simulators, and `X-IC-Tool` headers.
- **Runners (`bench/run_e2e.py`, `bench/run_bench.py`)**: orchestrate end-to-end evaluations, grid sweeps, and logging.
- **Reporting (`bench/report.py`)**: aggregates per-run summaries into CSV/Markdown artefacts.

## Data Flow
1. `tools.jsonl` and `servers.jsonl` feed the IC scorer to compute `ic_tool.jsonl` / `ic_server.json`.
2. The task generator consumes the tool catalogue and IC buckets to produce `tests/tasks/*.jsonl` suites.
3. The agent builds a retrieval index over tool cards, applying IC-aware penalties and slot policies driven by thresholds.
4. The server enforces schema validation, optional output checks, and interface wrappers (`none`, `flat`, `hard`).
5. `run_e2e.py` spins the server (if requested), iterates over task suites, and writes task-level telemetry plus aggregated metrics. `run_bench.py` sweeps model/condition grids.
6. `report.py` collects summaries under `runs/` and emits consolidated tables for analysis.

## Telemetry and Artefacts
- `analysis/out/`: IC feature dumps.
- `tests/tasks/`: smoke task suites for fuzz/generic pipelines.
- `runs/<name>/`: per-run JSONL results, summary JSON, and derived CSV/Markdown reports.
- Prompts under `prompts/` control system instructions, slot-filling behaviour, and validator feedback.
