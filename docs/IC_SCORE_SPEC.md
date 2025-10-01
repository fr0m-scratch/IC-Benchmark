# IC Score Specification

The scorer in `analysis/ic_score.py` emits feature vectors and aggregate scores for three scopes:

## Tool-Level (`IC_tool`)
Features extracted from the JSON Schema of each tool include:
- Structural counts: leaf/object/array fields, total/required fields, log-scaled leaf count.
- Nesting & control flow: maximum depth, union constructs (`oneOf`/`anyOf`/`allOf`/`if-then-else`), dependency operators.
- Domain constraints: enumerations, numeric ranges, array expectations, documentation token load, missing examples.

Weighted sum defaults are tuned heuristically and can be overridden with a JSON weights file. The CLI outputs one JSONL row per tool containing features, weight contributions, and the final score.

## Server-Level (`IC_server`)
Aggregates tool metrics and catalogue metadata:
- Tool count & log-scaled size.
- Mean `IC_tool`.
- Description ambiguity via pairwise cosine similarity.
- Naming inconsistency entropy for tool prefixes.
- Stateful hints (auth/rate/pagination) detected in descriptions.
- Documentation burden propagated from tool descriptions/server metadata.

Outputs a single JSON object keyed by `server_id`.

## Set-Level (`IC_set`)
Used at retrieval time to characterise candidate pools (Top-K):
- `log_k`, mean `IC_tool`, redundancy (cosine similarity of descriptions), token load, and server diversity.
- Exposed via `compute_ic_set` for in-run telemetry.

## Artefacts & Schema Contracts
- CLI: `python -m analysis.ic_score --tools tools.jsonl --servers servers.jsonl --out-dir analysis/out`.
- Outputs: `analysis/out/ic_tool.jsonl`, `analysis/out/ic_server.json`.
- JSON contract: `analysis/ic_features.jsonschema.json` covering tool/server/set entries.
