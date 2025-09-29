MCP Streaming Server and Agent (Python + Ollama/Gemini)

Overview
- Streamable HTTP MCP-like server that exposes tools loaded from a JSONL file and supports latency simulation.
- Streaming agent client for Ollama or Gemini: detects a tool call, executes it via the server (streamed), and returns the result to the model for a final answer.

Files
- `server_py/app.py` — Streamable HTTP server with validation and schema-derived placeholder outputs (no built-ins).
- `bench_py/agent_executor.py` — Production agent (plan → select → tool → retry → finalize). Supports Ollama and Gemini providers.
- `bench_py/agent.py` — Agent CLI entry; loads `.env`, selects provider (`--provider ollama|gemini`).
- `bench_py/evaluate.py` — Generic offline evaluator requiring `{tool,input}` per task; validates responses against `output_schema`.
- `bench_py/run_e2e.py` — End-to-end runner (server + offline eval + agent over tasks).
- `bench_py/run_bench.py` — Benchmark runner for both providers (tool-selection accuracy and latency stats).
- `bench/tools.sample.jsonl` — Minimal sample tool set.
- `bench/tools.advanced.jsonl` — Larger tool set for capability tests and demos.
- `tests/generic_tasks.jsonl` — Schema-valid tasks with explicit inputs (no preset logic).
- `tests/advanced_tasks.jsonl` — Prompt-based tasks with `expect.tool` for tool-selection evaluation.
- `servers.jsonl` — Real-world MCP server index fetched from public listings (e.g., Smithery). Reference data; not used directly by the programs.
- `tools.jsonl` — Real-world MCP tools catalog fetched from live MCP servers/indexes. Each line describes a tool with its schema and metadata.

Python Implementation
- `server_py/app.py` — Python streamable HTTP server with basic input validation and schema-derived outputs.
- `bench_py/agent_executor.py` — Production-style agent loop (plan → select → retrieve → tool-call; retries and fallback).
- `bench_py/agent.py` — CLI entry for the agent executor.
- `bench_py/run.py` — Simple Python client (earlier minimal harness, still usable).
- `bench_py/evaluate.py` — Generic offline evaluator: requires `tool` and `input` per task; validates outputs against `output_schema`.
- `bench_py/run_bench.py` — Provider benchmark harness (Ollama + Gemini) over a task suite.

Run the Server
- `python3 server_py/app.py --port 3001 --tools bench/tools.advanced.jsonl`
- Hot replace the JSONL any time; server reloads on request when file mtime changes.

Server Endpoints
- `GET /health` → `{ ok: true }`
- `GET /tools?limit=N&names=a,b` → JSON list (or NDJSON if `Accept: application/x-ndjson`).
- `POST /invoke?stream=1` with body `{ tool: string, input: object, latency_ms?: number }` → NDJSON stream of events: `start`, `progress` (optional), `result`, `end`.
  - If `stream` is not set, returns a single JSON: `{ ok: true, tool, output, latency_ms }`.
  - Output generation
    - If the tool has `output_schema`, server fills a placeholder object matching the schema and adds `echo` with the input.
    - Otherwise `{ echo: input }`.
  - Latency: uses `latency_ms` from request → tool → `--default-latency` → 0.

Run the Agent (Ollama)
- Requirements: Ollama running locally with a model, e.g. `ollama pull qwen2.5:7b-instruct`.
- Example single run:
  - `python3 server_py/app.py --port 3001 --tools bench/tools.advanced.jsonl`
  - `PYTHONPATH=. python3 bench_py/agent.py --model qwen2.5:7b-instruct --prompt "Add 2 and 3." --server http://localhost:3001 --ollama http://localhost:11434 --max-steps 3 --retry 1`

End-to-End Script
- One-shot E2E over a task suite with offline verification first:
  - `PYTHONPATH=. python3 bench_py/run_e2e.py --model qwen2.5:7b-instruct --port 3001 --tools bench/tools.advanced.jsonl --tasks tests/advanced_tasks.jsonl`
  - Output includes: server health, offline tool correctness, and online agent final-answer matches.

JSONL Tool Format
- Each line is a JSON object. Minimal fields:
  - `tool_name` (string)
  - `tool_description` (string)
  - `input_schema` (JSON Schema)
  - `output_schema` (JSON Schema, optional)
  - `latency_ms` (number, optional)

Notes
- This implementation focuses on streamable HTTP surfaces suitable for interface complexity experiments.
- Tools JSONL can be swapped anytime to vary interface complexity; per-tool latency supported.
- No Node artifacts; minimal Python-only stack.

Python Agent (Production Loop)
- Start Python server (choose a free port, e.g., 3001):
  - `python3 server_py/app.py --port 3001 --tools bench/tools.sample.jsonl`
- Run agent with Ollama (qwen2.5:7b-instruct):
  - `PYTHONPATH=. python3 bench_py/agent.py --model qwen2.5:7b-instruct --prompt "Add 2 and 3." --server http://localhost:3001 --ollama http://localhost:11434 --limit 5 --max-steps 3 --retry 1`
- The agent:
  - Plans briefly, asks model to either emit `TOOL_CALL {...}` or `FINAL_ANSWER:`.
  - Streams and executes tool calls with error handling and retries.
  - On success, feeds result back and requests `FINAL_ANSWER`.

Evaluate Tool Correctness (offline)
- With server running (same port as above):
  - `python3 bench_py/evaluate.py --server http://localhost:3001 --tasks tests/generic_tasks.jsonl`
  - Validates that outputs conform to each tool's `output_schema`.

Benchmark Both Providers (tool selection + latency)
- With server auto-started by the runner:
  - `PYTHONPATH=. python3 bench_py/run_bench.py --port 3015 --tools bench/tools.advanced.jsonl --tasks tests/advanced_tasks.jsonl --ollama-model qwen2.5:7b-instruct --gemini-model gemini-2.5-flash --limit 20 --max-steps 3 --retry 1`
- Prints JSON summaries for Ollama and Gemini with counts and latency stats.
Run the Agent (Gemini)
- Add your Gemini API key to `.env` as `GEMINI_API_KEY=...` (or export it in the environment).
- Example single run:
  - `python3 server_py/app.py --port 3001 --tools bench/tools.advanced.jsonl`
  - `PYTHONPATH=. python3 bench_py/agent.py --provider gemini --model gemini-1.5-flash --prompt "Add 2 and 3." --server http://localhost:3001 --limit 5 --max-steps 3 --retry 1`
  - Notes:
    - `.env` is auto-loaded; if `python-dotenv` is not installed, a lightweight loader is used.
    - You can also pass `--gemini-api-key` directly or rely on `GEMINI_API_KEY` in the environment.

## Preliminary Result (ICI + Retrieval, no training)
ICI-aware unsupervised reranking improves Recall@3 from ~0.83 to ~0.96 on the synthetic fuzzy set, proving interface complexity is a first-order factor for SLM-friendly retrieval.

Run:
- `bash scripts/run_prelim_ici.sh`

Expected metrics (λ∈[0.25,0.5]): Top-1≈0.71–0.79, Recall@3≈0.96.

## Gap Analysis Pipeline (E2–E4)
To compare SLM vs LLM performance across interface-complexity buckets:

1. Generate candidates (already created by `scripts/run_prelim_ici.sh`).
2. Run fixed-candidate evaluation for each model:
   - SLM (text-only):
     `python agent/e2e_eval.py --tasks data/prelim/tasks.jsonl --tools data/prelim/tools.jsonl --candidates data/prelim/results/candidates_text.jsonl --provider ollama --model <ollama-model> --outdir runs/slm_text --tag text`
   - LLM (text-only):
     `python agent/e2e_eval.py --tasks data/prelim/tasks.jsonl --tools data/prelim/tools.jsonl --candidates data/prelim/results/candidates_text.jsonl --provider gemini --model <gemini-model> --gemini-api-key $GEMINI_API_KEY --outdir runs/llm_text --tag text`
   - Repeat with `candidates_ici.jsonl` (`--tag ici`) for the ICI-aware reranked set.
   - For offline smoke tests, add `--mock-responses mock.jsonl` (contains `{"task_id":...,"response":"{...}"}`) to skip real model calls.
3. Analyze gaps and error types:
   `python scripts/analyze_gap.py --slm-text runs/slm_text/per_task_text.jsonl --llm-text runs/llm_text/per_task_text.jsonl --slm-ici runs/slm_ici/per_task_ici.jsonl --llm-ici runs/llm_ici/per_task_ici.jsonl --ici-features data/prelim/tools.features_ici.jsonl --outdir analysis/gap --figdir docs/figures`

Outputs include bucket-level tables (`analysis/gap/*.csv`) and figures (`docs/figures/success_vs_ici_*.png`, `gap_vs_ici.png`, `error_breakdown_*.png`) supporting Claims 2–4.

Dependencies:
- `python3 -m pip install matplotlib jsonschema`
