# Agent Specification

Implementation: `bench/agent_executor.py`

## Loop
1. **Plan**: log current context/intention.
2. **Retrieve**: vector search over tool cards (`bench/retrieval.py`) with IC-aware scoring `S* = sim - μ·IC_tool - ν·redundancy`.
3. **Gate**: compare adjusted score/gating threshold to decide direct answer vs tool invocation.
4. **Select**: take top candidate (multi-tool support can be extended) and compute slot policies based on IC.
5. **Argument Fill**:
   - Attempt structured decoding using provider output (`prompts/agent_system.txt`).
   - If invalid or high-IC, switch to slot-by-slot fills using `prompts/slot_filling.txt` and validator feedback.
6. **Invoke**: call the MCP server via `ToolInvoker`, streaming NDJSON when requested; emit telemetry and record latency.
7. **Repair**: on schema/server errors, regenerate minimal diffs guided by `prompts/validator_feedback.txt`.
8. **Finalize**: aggregate tool outputs into a final answer and expose telemetry + metrics.

## Providers (`bench/providers.py`)
- Unified interface for Ollama, Gemini, and Fireworks with JSON snipper extraction and retry behaviour.

## Arguments (`bench/arguments.py`)
- Schema descriptions, minimal argument synthesis, slot prompts, and validation helpers.

## Metrics (`bench/metrics.py`)
- Track invocation counts, pass@1, first-try validity, retries, latency, and IC bucket stratification.

## Configuration
- `AgentConfig` controls step limits, retry caps, IC penalties, slot thresholds, and temperature.
- YAML configs under `configs/` map directly to these field values and server launch options.

## Telemetry
- Per-step logs recorded in `AgentTelemetry` for downstream debugging and report generation.
- Metrics summary returned alongside the final answer; `run_e2e.py` aggregates across tasks.
