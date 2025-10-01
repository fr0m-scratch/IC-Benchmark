# Runbook

## Prerequisites
- Python 3.11+ environment with `requests`, `jsonschema`, and `pytest`.
- Optional: `pyyaml` for YAML configs, provider-specific SDKs (Gemini / Fireworks) if using cloud models.
- Set environment variables in `.env` (load with `dotenv` or export):
  - `GEMINI_API_KEY`
  - `FIREWORKS_API_KEY`
  - `OLLAMA_HOST` (e.g., `http://localhost:11434`)

## Quick Start
1. **Install deps** (example):
   ```bash
   pip install -r requirements.txt  # if provided
   ```
2. **Generate tasks**:
   ```bash
   make tasks
   ```
3. **Smoke test** (local Ollama):
   ```bash
   make e2e
   ```
4. **Benchmark grid**:
   ```bash
   make bench
   ```
5. **Reports**:
   ```bash
   make report
   ```

## Server Controls
- Launch manually:
  ```bash
  python server/app.py --port 3001 --tools tools.jsonl --wrap flat --validate-output
  ```
- Flags:
  - `--wrap {none,flat,hard}` to switch schema variants.
  - `--require-auth --auth-key KEY` to demand headers.
  - `--rate-limit N`, `--paginate --page-size K` to simulate server-level IC.

## Debugging Tips
- IC artefacts: `analysis/out/`.
- Task suites: `tests/tasks/*.jsonl`.
- Agent telemetry per task: `runs/<run>/results.jsonl`.
- Bench grid summary: `runs/ablation/bench_summary.json`.
