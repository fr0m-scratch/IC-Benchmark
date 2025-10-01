# Task Generator Specification

The generator (`bench/taskgen.py`) builds benchmark suites in the style of MCP-Bench.

## Inputs
- `tools.jsonl`: catalogue of tools with names, descriptions, schemas, and metadata.
- Optional CLI flags: tiers, seed, output paths, limit.

## Pipeline
1. **Normalisation**: convert tool entries into `ToolRecord`s and cache `IC_tool` scores.
2. **Dependency Mining**: inspect required fields and descriptions to infer upstream tools (e.g., fields ending in `_id` linked to `list/get` tools).
3. **Schema Sampling**: sample arguments via `SchemaSampler`, respecting numeric ranges, enums, formats, and optional field policies by tier.
4. **Instruction Synthesis**:
   - `generic_tasks.jsonl`: includes `tool` and JSON `input` payload ready for offline validation.
   - `fuzzy_tasks.jsonl`: natural-language instructions with obfuscated tool hints plus expected tool/keys metadata.
5. **IC Bucketing**: each task stores the originating tool IC and bucket (`low`/`medium`/`high`) for downstream stratification.

## CLI
```
python -m bench.taskgen --tools tools.jsonl \
    --generic-out tests/tasks/generic_tasks.jsonl \
    --fuzzy-out tests/tasks/fuzzy_tasks.jsonl \
    --tiers realistic,hard --seed 13 --limit 100
```

## Outputs
- `tests/tasks/generic_tasks.jsonl`: smoke set with explicit inputs.
- `tests/tasks/fuzzy_tasks.jsonl`: fuzzed instructions with hidden expectations.
- Unit tests (`tests/test_taskgen.py`) validate dependency inference and sampling.
