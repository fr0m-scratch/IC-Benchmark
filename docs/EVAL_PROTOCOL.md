# Evaluation Protocol

## Experiment Conditions
- **BASE**: retrieval penalties disabled, original schemas, simple argument fill.
- **+RET**: enable IC-aware retrieval penalties (μ, ν) while keeping original schemas.
- **+WRAP**: serve flat-easy schema wrappers for high-IC tools via `--wrap flat` on the server.
- **+SLOT**: force slot-by-slot argument filling for high-IC tools (`slot_threshold=0`).
- **FULL**: combine +RET, +WRAP, and +SLOT.

## Models
- SLM baseline (e.g., Ollama `qwen2.5:7b-instruct`).
- Cloud LLM via Gemini or Fireworks. Configurable through `grid.models` in `configs/ablation.yaml`.

## Metrics
Collected per task and aggregated in `summary.json` / bench summaries:
- `Pass@1`, `FirstTryValid`, `Pass@1` by IC buckets.
- Invocation counts, retries per solved task.
- Latency averages.

`run_bench.py` writes `bench_summary.json` summarising each model/condition pair. `bench/report.py` converts results into CSV/Markdown for paper-ready tables.

## Procedure
1. Generate task suites (if needed): `make tasks`.
2. Run smoke SLM experiment: `make e2e`.
3. Execute grid: `make bench` (reads `configs/ablation.yaml`).
4. Aggregate reports: `make report`.
