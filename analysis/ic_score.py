"""Interface Complexity (IC) scoring utilities.

This module provides tooling to compute Interface Complexity metrics at three
levels:

* Tool level (`IC_tool`): derived from the JSON Schema of each tool's inputs
  (and outputs when available).
* Server level (`IC_server`): aggregates tool-level signals and server
  metadata to capture catalogue-level ambiguity and operational overhead.
* Set level (`IC_set`): characterises the complexity of a retrieved candidate
  set (Top-K tools) used during agent retrieval/reranking.

The implementation follows the high-level specification in
``docs/IC_SCORE_SPEC.md``. The goal is to produce deterministic feature vectors
that can be consumed by downstream components (retrieval rerankers, telemetry,
reports) while remaining lightweight enough to run offline as part of the
benchmark tooling.

Typical usage (CLI):

```
python -m analysis.ic_score --tools tools.jsonl --servers servers.jsonl \
    --out-dir analysis/out
```

The CLI emits two artefacts by default:

* ``ic_tool.jsonl`` – one JSON record per tool with feature vector and score.
* ``ic_server.json`` – server-level aggregation keyed by server id.

In addition, the module exposes ``compute_ic_set`` which can be used at runtime
by the agent to measure the complexity of a candidate pool.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Data classes and lightweight records
# ---------------------------------------------------------------------------


@dataclass
class ToolRecord:
    """Normalised view of a tool card loaded from JSONL."""

    tool_id: str
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Optional[Dict[str, Any]]
    server_id: str
    server_name: str
    raw: Dict[str, Any]


@dataclass
class ToolICResult:
    """Feature vector and score for a tool."""

    tool: ToolRecord
    features: Dict[str, float]
    score: float
    components: Dict[str, float]


@dataclass
class ServerICResult:
    server_id: str
    server_name: str
    features: Dict[str, float]
    score: float
    components: Dict[str, float]
    tool_ids: List[str]


# ---------------------------------------------------------------------------
# Schema traversal helpers
# ---------------------------------------------------------------------------


@dataclass
class SchemaMetrics:
    """Intermediate accumulators for schema analytics."""

    leaf_fields: int = 0
    object_fields: int = 0
    array_fields: int = 0
    total_fields: int = 0
    required_fields: int = 0
    max_depth: int = 0
    union_ops: int = 0
    dependency_ops: int = 0
    enum_bits: float = 0.0
    numeric_bits: float = 0.0
    string_burden: float = 0.0
    array_cost: float = 0.0
    doc_tokens: int = 0
    missing_examples: int = 0


PRIMITIVE_TYPES = {"string", "number", "integer", "boolean", "null"}
UNION_KEYS = ("oneOf", "anyOf", "allOf")
DEPENDENCY_KEYS = ("dependencies", "dependentRequired", "dependentSchemas")
DESC_KEYS = ("description", "title", "summary")


def _as_iter(value: Any) -> Iterable[Any]:
    if isinstance(value, (list, tuple)):
        return value
    return (value,)


def _tokenise(text: str) -> List[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [tok for tok in cleaned.split() if tok]


def _coerce_number(schema: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        value = schema.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


class SchemaAnalyser:
    """Walks a JSON Schema and extracts structural metrics."""

    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema or {}
        self.metrics = SchemaMetrics()

    def analyse(self) -> SchemaMetrics:
        self._walk(self.schema, depth=1)
        return self.metrics

    # Recursive traversal -------------------------------------------------

    def _walk(self, schema: Any, depth: int) -> None:
        if not isinstance(schema, MutableMapping):
            return
        self.metrics.max_depth = max(self.metrics.max_depth, depth)

        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            # Prefer non-null concrete type for traversal heuristics
            non_null = [t for t in schema_type if t != "null"]
            schema_type = non_null[0] if non_null else schema_type[0]

        # Count textual documentation burden
        for key in DESC_KEYS:
            text = schema.get(key)
            if isinstance(text, str) and text.strip():
                tokens = _tokenise(text)
                self.metrics.doc_tokens += len(tokens)
                # Encourage documentation but penalise extremely long strings.
                self.metrics.string_burden += len(tokens) * 0.05

        examples = schema.get("examples")
        if isinstance(examples, list) and examples:
            for ex in examples:
                if isinstance(ex, str):
                    tokens = _tokenise(ex)
                    self.metrics.doc_tokens += len(tokens)
                    self.metrics.string_burden += len(tokens) * 0.02
        else:
            # Missing examples make slot-filling harder.
            self.metrics.missing_examples += 1
            self.metrics.string_burden += 0.25

        # Union-like constructs propagate depth
        for key in UNION_KEYS:
            variants = schema.get(key)
            if isinstance(variants, list):
                self.metrics.union_ops += len(variants)
                for subschema in variants:
                    self._walk(subschema, depth + 1)

        if "if" in schema:
            self.metrics.union_ops += 1
            self._walk(schema.get("if"), depth + 1)
            if schema.get("then"):
                self._walk(schema.get("then"), depth + 1)
            if schema.get("else"):
                self._walk(schema.get("else"), depth + 1)

        for key in DEPENDENCY_KEYS:
            dep_val = schema.get(key)
            if isinstance(dep_val, dict):
                self.metrics.dependency_ops += len(dep_val)

        if schema_type == "object" or (schema_type is None and schema.get("properties")):
            self.metrics.object_fields += 1
            properties = schema.get("properties")
            required = set(schema.get("required") or [])
            if isinstance(properties, dict):
                for prop_name, subschema in properties.items():
                    self.metrics.total_fields += 1
                    if prop_name in required:
                        self.metrics.required_fields += 1
                    self._walk(subschema, depth + 1)

        elif schema_type == "array":
            self.metrics.array_fields += 1
            min_items = schema.get("minItems")
            max_items = schema.get("maxItems")
            if isinstance(min_items, int) and isinstance(max_items, int) and max_items >= min_items:
                expected_len = (min_items + max_items) / 2.0
            elif isinstance(min_items, int):
                expected_len = max(float(min_items), 1.0)
            elif isinstance(max_items, int):
                expected_len = max(float(max_items) * 0.5, 1.0)
            else:
                expected_len = 1.5  # heuristic default
            self.metrics.array_cost += expected_len

            items_schema = schema.get("items")
            if isinstance(items_schema, list):
                for subschema in items_schema:
                    self._walk(subschema, depth + 1)
            else:
                self._walk(items_schema, depth + 1)

        elif schema_type in PRIMITIVE_TYPES or schema_type is None:
            self.metrics.leaf_fields += 1
            # Enums contribute to combinatorial burden.
            enum_values = schema.get("enum")
            if isinstance(enum_values, list) and enum_values:
                self.metrics.enum_bits += math.log2(len(enum_values))

            if schema_type in {"number", "integer"}:
                minimum = _coerce_number(schema, ["minimum", "exclusiveMinimum"])
                maximum = _coerce_number(schema, ["maximum", "exclusiveMaximum"])
                multiple_of = _coerce_number(schema, ["multipleOf"])
                if multiple_of is None:
                    multiple_of = 1.0 if schema_type == "integer" else 0.1
                if minimum is not None and maximum is not None and maximum > minimum:
                    span = maximum - minimum
                    buckets = span / multiple_of
                    self.metrics.numeric_bits += math.log2(max(buckets, 1.0) + 1.0)
                elif minimum is not None or maximum is not None:
                    span = abs((maximum or 0.0) - (minimum or 0.0))
                    self.metrics.numeric_bits += math.log2(span / multiple_of + 2.0)
                else:
                    # Open range – assume moderate burden.
                    self.metrics.numeric_bits += math.log2(20.0 / multiple_of)

            if schema_type == "string":
                format_hint = schema.get("format")
                if isinstance(format_hint, str) and format_hint:
                    self.metrics.string_burden += 0.5  # extra constraint


# ---------------------------------------------------------------------------
# Tool-level scoring
# ---------------------------------------------------------------------------


DEFAULT_TOOL_WEIGHTS: Dict[str, float] = {
    "log_leaf_fields": 0.8,
    "required_ratio": 0.6,
    "max_depth": 0.7,
    "union_ops": 0.5,
    "dependency_ops": 0.4,
    "enum_bits": 0.45,
    "numeric_bits": 0.4,
    "string_burden": 0.35,
    "array_cost": 0.3,
    "missing_examples": 0.25,
}


def _compute_tool_features(schema_metrics: SchemaMetrics) -> Dict[str, float]:
    total_fields = max(schema_metrics.total_fields, 1)
    features = {
        "leaf_fields": float(schema_metrics.leaf_fields),
        "object_fields": float(schema_metrics.object_fields),
        "array_fields": float(schema_metrics.array_fields),
        "total_fields": float(total_fields),
        "required_fields": float(schema_metrics.required_fields),
        "required_ratio": float(schema_metrics.required_fields) / float(total_fields),
        "max_depth": float(schema_metrics.max_depth),
        "union_ops": float(schema_metrics.union_ops),
        "dependency_ops": float(schema_metrics.dependency_ops),
        "enum_bits": float(schema_metrics.enum_bits),
        "numeric_bits": float(schema_metrics.numeric_bits),
        "string_burden": float(schema_metrics.string_burden),
        "array_cost": float(schema_metrics.array_cost),
        "doc_tokens": float(schema_metrics.doc_tokens),
        "missing_examples": float(schema_metrics.missing_examples),
    }
    features["log_leaf_fields"] = math.log2(features["leaf_fields"] + 1.0)
    return features


def score_tool(tool: ToolRecord, weights: Optional[Dict[str, float]] = None) -> ToolICResult:
    weights = {**DEFAULT_TOOL_WEIGHTS, **(weights or {})}
    analyser = SchemaAnalyser(tool.input_schema or {})
    metrics = analyser.analyse()
    features = _compute_tool_features(metrics)
    components: Dict[str, float] = {}
    for key, weight in weights.items():
        value = features.get(key, 0.0)
        components[key] = weight * value
    score = sum(components.values())
    return ToolICResult(tool=tool, features=features, score=score, components=components)


# ---------------------------------------------------------------------------
# Server-level scoring
# ---------------------------------------------------------------------------


DEFAULT_SERVER_WEIGHTS: Dict[str, float] = {
    "log_tool_count": 0.6,
    "mean_ic_tool": 0.8,
    "ambiguity": 0.7,
    "naming_inconsistency": 0.5,
    "statefulness": 0.4,
    "doc_burden": 0.35,
}

STATEFUL_HINTS = ("auth", "token", "key", "login", "paginate", "cursor", "offset", "rate", "quota")


def _pairwise_cosine(texts: Sequence[str]) -> float:
    if len(texts) < 2:
        return 0.0
    vectors: List[Counter[str]] = []
    for text in texts:
        tokens = _tokenise(text)
        vectors.append(Counter(tokens))
    norms: List[float] = [math.sqrt(sum(v * v for v in vec.values())) for vec in vectors]
    dots = 0.0
    pairs = 0
    for i, vec in enumerate(vectors):
        norm_i = norms[i]
        for j in range(i + 1, len(vectors)):
            norm_j = norms[j]
            if norm_i == 0 or norm_j == 0:
                continue
            dot = sum(vec[t] * vectors[j].get(t, 0) for t in vec)
            dots += dot / (norm_i * norm_j)
            pairs += 1
    return dots / pairs if pairs else 0.0


def score_server(
    server_id: str,
    server_name: str,
    tool_results: Sequence[ToolICResult],
    server_meta: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> ServerICResult:
    weights = {**DEFAULT_SERVER_WEIGHTS, **(weights or {})}
    tool_count = len(tool_results)
    tool_scores = [tr.score for tr in tool_results]
    mean_ic = statistics.fmean(tool_scores) if tool_scores else 0.0
    descriptions = [tr.tool.description for tr in tool_results if tr.tool.description]
    ambiguity = _pairwise_cosine(descriptions)

    # Naming inconsistency: measure entropy of prefix tokens.
    prefixes = []
    for tr in tool_results:
        name = tr.tool.name
        prefixes.append(name.split("_")[0].split(" ")[0].lower())
    counter = Counter(prefixes)
    total = sum(counter.values()) or 1
    entropy = -sum((count / total) * math.log(count / total + 1e-9) for count in counter.values())
    naming_inconsistency = entropy

    statefulness = 0
    for text in descriptions:
        if any(hint in text.lower() for hint in STATEFUL_HINTS):
            statefulness += 1
    if server_meta:
        desc = str(server_meta.get("description", ""))
        if any(hint in desc.lower() for hint in STATEFUL_HINTS):
            statefulness += 1
    statefulness = float(statefulness) / max(tool_count, 1)

    doc_tokens = 0
    if server_meta:
        for key in DESC_KEYS:
            text = server_meta.get(key)
            if isinstance(text, str):
                doc_tokens += len(_tokenise(text))
    doc_tokens += statistics.fmean([tr.features.get("doc_tokens", 0.0) for tr in tool_results]) if tool_results else 0.0

    features = {
        "tool_count": float(tool_count),
        "log_tool_count": math.log2(tool_count + 1.0),
        "mean_ic_tool": float(mean_ic),
        "ambiguity": float(ambiguity),
        "naming_inconsistency": float(naming_inconsistency),
        "statefulness": float(statefulness),
        "doc_burden": float(doc_tokens * 0.05),
    }

    components = {key: weights.get(key, 0.0) * value for key, value in features.items() if key in weights}
    score = sum(components.values())
    return ServerICResult(
        server_id=server_id,
        server_name=server_name,
        features=features,
        score=score,
        components=components,
        tool_ids=[tr.tool.tool_id for tr in tool_results],
    )


# ---------------------------------------------------------------------------
# Set-level scoring
# ---------------------------------------------------------------------------


DEFAULT_SET_WEIGHTS: Dict[str, float] = {
    "log_k": 0.4,
    "mean_ic_tool": 0.6,
    "redundancy": 0.7,
    "token_load": 0.45,
    "server_diversity": 0.3,
}


@dataclass
class SetICResult:
    tools: List[str]
    features: Dict[str, float]
    score: float
    components: Dict[str, float]


def compute_ic_set(
    tool_results: Sequence[ToolICResult],
    weights: Optional[Dict[str, float]] = None,
) -> SetICResult:
    weights = {**DEFAULT_SET_WEIGHTS, **(weights or {})}
    k = len(tool_results)
    if k == 0:
        return SetICResult(tools=[], features={"k": 0.0}, score=0.0, components={})

    descriptions = [tr.tool.description for tr in tool_results]
    redundancy = _pairwise_cosine(descriptions)
    token_load = sum(len(_tokenise(desc)) for desc in descriptions)
    mean_ic = statistics.fmean([tr.score for tr in tool_results]) if tool_results else 0.0
    server_ids = [tr.tool.server_id for tr in tool_results]
    unique_servers = len(set(server_ids))
    server_diversity = unique_servers / k

    features = {
        "k": float(k),
        "log_k": math.log2(k + 1.0),
        "redundancy": float(redundancy),
        "token_load": float(token_load),
        "mean_ic_tool": float(mean_ic),
        "server_diversity": float(server_diversity),
    }

    components = {key: weights.get(key, 0.0) * value for key, value in features.items() if key in weights}
    score = sum(components.values())
    return SetICResult(
        tools=[tr.tool.tool_id for tr in tool_results],
        features=features,
        score=score,
        components=components,
    )


# ---------------------------------------------------------------------------
# Loading utilities and CLI wiring
# ---------------------------------------------------------------------------


def _normalise_tool(raw: Dict[str, Any]) -> Optional[ToolRecord]:
    input_schema = raw.get("input_schema") or raw.get("inputSchema")
    if not isinstance(input_schema, dict):
        return None
    server_id = str(raw.get("server_id") or raw.get("serverId") or raw.get("server") or "")
    server_name = str(raw.get("server_name") or raw.get("serverName") or server_id)
    name = str(raw.get("tool_name") or raw.get("name") or raw.get("id") or "").strip()
    if not name:
        return None
    description = str(raw.get("tool_description") or raw.get("description") or "").strip()
    tool_id = raw.get("tool_id")
    if not tool_id:
        prefix = server_id or raw.get("server") or "anon"
        tool_id = f"{prefix}::{name}"
    return ToolRecord(
        tool_id=str(tool_id),
        name=name,
        description=description,
        input_schema=input_schema,
        output_schema=raw.get("output_schema") if isinstance(raw.get("output_schema"), dict) else None,
        server_id=server_id or "unknown_server",
        server_name=server_name or (server_id or "unknown_server"),
        raw=raw,
    )


def load_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_tools(path: pathlib.Path) -> List[ToolRecord]:
    records = []
    for raw in load_jsonl(path):
        tool = _normalise_tool(raw)
        if tool:
            records.append(tool)
    return records


def load_servers(path: pathlib.Path) -> Dict[str, Dict[str, Any]]:
    servers: Dict[str, Dict[str, Any]] = {}
    for raw in load_jsonl(path):
        server_id = str(raw.get("id") or raw.get("server_id") or raw.get("qualifiedName") or "")
        if not server_id:
            continue
        servers[server_id] = raw
    return servers


def compute_all_tool_scores(
    tools: Sequence[ToolRecord],
    weights: Optional[Dict[str, float]] = None,
) -> List[ToolICResult]:
    return [score_tool(tool, weights=weights) for tool in tools]


def compute_all_server_scores(
    tool_results: Sequence[ToolICResult],
    server_meta: Optional[Dict[str, Dict[str, Any]]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> List[ServerICResult]:
    by_server: Dict[str, List[ToolICResult]] = defaultdict(list)
    for result in tool_results:
        by_server[result.tool.server_id].append(result)

    server_scores: List[ServerICResult] = []
    for server_id, results in by_server.items():
        meta = server_meta.get(server_id) if server_meta else None
        server_scores.append(
            score_server(
                server_id=server_id,
                server_name=results[0].tool.server_name,
                tool_results=results,
                server_meta=meta,
                weights=weights,
            )
        )
    return server_scores


def _dump_tool_scores(results: Sequence[ToolICResult], out_path: pathlib.Path) -> None:
    with out_path.open("w", encoding="utf-8") as handle:
        for result in results:
            record = {
                "scope": "tool",
                "tool_id": result.tool.tool_id,
                "server_id": result.tool.server_id,
                "features": result.features,
                "components": result.components,
                "score": result.score,
                "metadata": {
                    "name": result.tool.name,
                    "description": result.tool.description,
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _dump_server_scores(results: Sequence[ServerICResult], out_path: pathlib.Path) -> None:
    serialised = {
        result.server_id: {
            "scope": "server",
            "server_id": result.server_id,
            "server_name": result.server_name,
            "features": result.features,
            "components": result.components,
            "score": result.score,
            "tool_ids": result.tool_ids,
        }
        for result in results
    }
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(serialised, handle, ensure_ascii=False, indent=2)


def _load_weights(path: Optional[pathlib.Path]) -> Optional[Dict[str, float]]:
    if not path:
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Weights file must contain a JSON object")
    return {str(k): float(v) for k, v in data.items()}


def run_cli(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Compute Interface Complexity scores")
    parser.add_argument("--tools", type=pathlib.Path, required=True, help="Path to tools JSONL catalogue")
    parser.add_argument("--servers", type=pathlib.Path, help="Optional servers JSONL for metadata")
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("analysis/out"))
    parser.add_argument("--tool-weights", type=pathlib.Path, help="Optional JSON file overriding tool weights")
    parser.add_argument("--server-weights", type=pathlib.Path, help="Optional JSON file overriding server weights")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    tool_weights = _load_weights(args.tool_weights)
    server_weights = _load_weights(args.server_weights)

    tools = load_tools(args.tools)
    if not tools:
        raise SystemExit("No valid tools found in catalogue")

    tool_results = compute_all_tool_scores(tools, weights=tool_weights)
    _dump_tool_scores(tool_results, args.out_dir / "ic_tool.jsonl")

    servers_meta = load_servers(args.servers) if args.servers else None
    server_results = compute_all_server_scores(tool_results, server_meta=servers_meta, weights=server_weights)
    _dump_server_scores(server_results, args.out_dir / "ic_server.json")

    # Summary for humans running the CLI
    mean_ic_tool = statistics.fmean([result.score for result in tool_results])
    max_ic_tool = max(result.score for result in tool_results)
    mean_ic_server = statistics.fmean([result.score for result in server_results]) if server_results else 0.0
    print(f"Computed IC_tool for {len(tool_results)} tools (mean={mean_ic_tool:.3f}, max={max_ic_tool:.3f})")
    if server_results:
        print(f"Computed IC_server for {len(server_results)} servers (mean={mean_ic_server:.3f})")


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
