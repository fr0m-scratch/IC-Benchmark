# MCP Agent Benchmark

Tools and experiments for studying how interface complexity (IC) affects tool-using agents.

- MCP-style HTTP server that serves tool catalogs (with optional schema wrappers).
- Python agent harness + runners for single prompts, end-to-end suites, and grid sweeps.
- Analysis utilities for IC scoring, task generation, and reporting.

Note: `archive/` contains legacy snapshots and is not part of the active flow.

## Repository Layout

- `analysis/` — IC scoring utilities.
  - `ic_score.py` computes IC metrics for tools, servers, and candidate sets (CLI-friendly).
  - `ic_features.jsonschema.json` defines the emitted feature vectors.
  - `out/` holds cached scoring artefacts from the CLI (optional).
- `bench/` — Agent harness and benchmarking scripts.
  - `agent_executor.py` implements the complexity-aware loop: plan → retrieve → gate → fill → invoke → repair → finalize.
  - `agent.py` is a single-prompt CLI for a running MCP server.
  - `run_e2e.py` runs one experiment from a YAML/JSON config and writes results.
  - `run_bench.py` expands a grid of models × conditions and aggregates summaries.
  - `taskgen.py` synthesises generic and fuzzy task suites from a tool catalogue.
  - Support: `arguments.py`, `metrics.py`, `providers.py`, `retrieval.py`, `report.py`.
- `configs/` — Example YAML configs (e.g., `local_qwen.yaml`, `ablation.yaml`).
- `docs/` — Specs and design notes (`AGENT_SPEC.md`, `IC_SCORE_SPEC.md`, `TASK_GENERATOR_SPEC.md`, `EVAL_PROTOCOL.md`, etc.).
- `prompts/` — Prompt templates (system, slot-filling, validator).
- `server/` — MCP-style HTTP server.
  - `app.py` exposes `/health`, `/tools`, `/invoke`, and supports auth/rate/pagination toggles.
  - `ic_wrap.py` defines schema wrappers (`none`, `flat`, `hard`).
- `tests/` — Pytest suite for scoring, taskgen, agent, providers, retrieval, and runners.
- `runs/` — Experiment outputs (per-task logs and summaries, bench reports).
- `tools.jsonl` / `servers.jsonl` — MCP catalogues used by the server, analysis, and taskgen.
- `Makefile` — Convenience targets for common workflows.
- `Runbook` — Step-by-step instructions (more detailed than this README).

## Quick Start

1) Start the server

```bash
python3 server/app.py --port 3001 --tools tools.jsonl --wrap none
```

2) Generate tasks

```bash
make tasks
# writes tests/generic_tasks.jsonl and tests/fuzzy_tasks.jsonl
```

3) Single prompt sanity check (optional)

```bash
PYTHONPATH=. python3 bench/agent.py --provider ollama --model qwen2.5:7b-instruct \
  --server http://localhost:3001 --ollama http://localhost:11434 \
  --prompt "Add 2 and 3." --max-steps 3 --retry 1
```

4) End-to-end evaluation

```bash
PYTHONPATH=. python3 bench/run_e2e.py --config configs/local_qwen.yaml
```

5) Sweep multiple conditions

```bash
PYTHONPATH=. python3 bench/run_bench.py --config configs/ablation.yaml
```

6) Aggregate reports

```bash
make report
```

## Make Targets

- `make server` — run the HTTP server with the local catalogue.
- `make tasks` — generate `tests/generic_tasks.jsonl` and `tests/fuzzy_tasks.jsonl`.
- `make e2e` — run a single experiment (`configs/local_qwen.yaml`).
- `make bench` — run the grid benchmark (`configs/ablation.yaml`).
- `make report` — collate summaries under `runs/` to CSV/Markdown.
- `make tests` — run unit tests.

## Providers and Credentials

- Ollama: set `OLLAMA_HOST` (default `http://localhost:11434`). Ensure model availability with `ollama pull qwen2.5:7b-instruct`.
- Fireworks: set `FIREWORKS_API_KEY`.
- Gemini: set `GEMINI_API_KEY`.

Providers are auto-filtered in `bench/run_bench.py` based on availability/credentials to avoid failing runs.
