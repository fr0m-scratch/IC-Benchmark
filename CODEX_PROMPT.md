# IC-Benchmark: Full E2E Implementation Plan for Code Agent

You are a senior engineer tasked to finish and harden this repository so we can run the end-to-end experiments for our WWW’26 paper on **Interface Complexity (IC)** in MCP-style tool use.

## Context (current repo state)
- Streamable HTTP MCP-like server with:
  - `GET /health`, `GET /tools?limit=&names=`, `POST /invoke?stream=1` (NDJSON: start/progress/result/end). 
  - Outputs placeholder objects from `output_schema` or `{echo: input}`; optional latency simulation.  
- Python agent harness + runners:
  - `bench/agent_executor.py` (plan → select → call → retry → finalize)
  - `bench/agent.py` (CLI)
  - `bench/run_e2e.py`, `bench/run_bench.py`
- Data: `tools.jsonl`, `servers.jsonl` (large MCP catalogs).  
(*See repo README section in this branch.*)
  
## Objective
Ship a production-quality, reproducible E2E system that:
1. **Computes IC-Score** at three layers: Tool / Server / Set (candidate pool).
2. **Generates tasks** automatically from tool schemas (MCP-Bench style), including obfuscation and dependency chains.
3. **Implements a complete best-practice Agent** (retrieval → gate → selection → argument fill with constrained decoding → call → repair → multi-step planning → finalize) with **complexity-aware** variants.
4. **Runs controlled A/B experiments** that demonstrate: SLMs are more sensitive to interface complexity, and making any one stage complexity-aware **narrows the SLM/LLM gap**.
5. **Produces paper-ready reports** (tables/plots & CSV/JSON logs).

## Deliverables (create/update files)

### A) IC Score & Data Contracts
- [ ] `analysis/ic_score.py`: Compute `IC_tool`, `IC_server`, `IC_set` (see `docs/IC_SCORE_SPEC.md`).
- [ ] `analysis/ic_features.jsonschema.json`: JSON Schema for IC feature vectors.
- [ ] Wire IC scores into agent retrieval/rerank and logs.

### B) Task Generator (MCP-Bench–style)
- [ ] `bench/taskgen.py`: 
  - Parse `tools.jsonl`, infer tool dependencies from `input_schema`/desc.
  - Synthesize tasks → fuzz/obfuscate surface forms (hide tool names, keep constraints).
  - Two suites: `generic_tasks.jsonl` (explicit fields) and `fuzzy_tasks.jsonl` (NL-only with expected tool).
  - Reference: Obfuscation + dependency chain construction like MCP-Bench tasks. 
- [ ] `tests/tasks/` with small smoke sets.

### C) Agent (complete E2E with best practices)
- [ ] `bench/agent_executor.py`:
  - Add **tool retrieval** (vector index over tool cards), **complexity-aware rerank**: `S* = sim - μ·IC_tool - ν·redundancy`.
  - Add **tool-use gate** (decide answer-vs-tool).
  - Add **selection** (single or parallel).
  - Add **argument fill** with **constrained decoding** (JSON Schema), fallback to **slot-by-slot fill + per-slot validation** for high-IC tools.
  - Add **error taxonomy & repair** (limited retries, smallest-delta fix).
  - Add **multi-step planning** (ReAct-style scratchpad) across servers.
  - Log every decision + failure type.
- [ ] `bench/providers.py`: unify **Ollama**, **Gemini**, **Fireworks** chat providers via a single interface.
- [ ] `bench/retrieval.py`: FAISS/Chroma/Embeddings wrapper (local).
- [ ] `bench/metrics.py`: compute Pass@N, FirstTryValid, ToolSelectAcc, Steps, Switches, Token/Latency, and stratify by IC buckets.
- [ ] Prompts in `prompts/` (system + tool-use + slot-filling templates).

### D) Server upgrades
- [ ] `server/app.py`:
  - Validate request against `input_schema`, optionally against `output_schema`.
  - Add toggles: **complexity wrappers** (flat-easy vs original vs hard) per tool.
  - Add pagination/rate-limit/auth simulators (to model `IC_server`).
  - Hot-reload tools JSONL by mtime (already noted) and emit `x-ic-tool` headers with current scores.
- [ ] `server/ic_wrap.py`: wrappers to project complex schemas into simpler “narrow” forms (defaults/examples filled, optional two-step).

### E) Runners & Reports
- [ ] `bench/run_e2e.py`: spins server + runs offline tool validation + agent over tasks + aggregates metrics.
- [ ] `bench/run_bench.py`: grid over (model × condition) with seeds; saves CSV/JSON.
- [ ] `bench/report.py`: generate plots/tables for the paper.
- [ ] `Makefile` + `configs/*.yaml` to parameterize experiments.

### F) Docs
- [ ] `docs/AGENT_SPEC.md`, `docs/TASK_GENERATOR_SPEC.md`, `docs/IC_SCORE_SPEC.md`, `docs/EVAL_PROTOCOL.md`, `docs/RUNBOOK.md`, `docs/ARCHITECTURE.md`.

## Acceptance Criteria
- Reproducible runs on local machine with **Ollama** (SLM) and cloud LLM (Gemini/Fireworks).
- Passes smoke tests; produces CSV/JSON and plots with **SLM vs LLM gap** that narrows when turning on any single complexity-aware switch (retrieval OR wrapper OR slot-by-slot).
- Code style: Python 3.11, type hints, unit tests for IC-score + taskgen + validators.

## Implementation Order (must follow)
1) IC scoring (`analysis/ic_score.py`)  
2) Task generator (`bench/taskgen.py`)  
3) Agent completeness (retrieval/gate/selection/arguments/repair/planning)  
4) Server complexity toggles & validators  
5) Runners & metrics & report  
6) Docs + Makefile + configs

## Commands (expected to work)
See `docs/RUNBOOK.md`. Smoke:
```bash
python3 server/app.py --port 3001 --tools bench/tools.advanced.jsonl
PYTHONPATH=. python3 bench/agent.py --model qwen2.5:7b-instruct \
  --server http://localhost:3001 --ollama http://localhost:11434 \
  --prompt "Add 2 and 3." --max-steps 3 --retry 1

# End-to-end:
PYTHONPATH=. python3 bench/run_e2e.py --config configs/local_qwen.yaml
