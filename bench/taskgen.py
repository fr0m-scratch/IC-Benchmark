"""Task synthesis utilities for MCP-style tool benchmarks.

The generator consumes a tools catalogue (JSONL) and produces two suites:

* ``generic_tasks.jsonl``: explicit ``{tool, input}`` payloads that can be used
  for offline validation.
* ``fuzzy_tasks.jsonl``: natural-language only instructions with hidden ground
  truth used for end-to-end evaluations.

Generation draws inspiration from MCP-Bench:

* Infer light-weight dependency chains from schema semantics.
* Sample arguments that satisfy JSON Schema constraints.
* Obfuscate tool names/descriptions so that the agent must reason about
  capabilities rather than identifiers.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from analysis.ic_score import (
    ToolICResult,
    ToolRecord,
    compute_all_tool_scores,
    load_tools,
)

# ---------------------------------------------------------------------------
# Helpers for text normalisation and sampling
# ---------------------------------------------------------------------------


_WORD_BREAK = re.compile(r"[^a-z0-9]+")


def _tokenise(text: str) -> List[str]:
    return [tok for tok in _WORD_BREAK.split(text.lower()) if tok]


def _slug(path_segments: Sequence[str]) -> str:
    joined = "_".join(seg for seg in path_segments if seg)
    return re.sub(r"[^a-z0-9_]+", "_", joined.lower()) or "value"


def _strip_tool_mentions(description: str, tool_name: str) -> str:
    pattern = re.escape(tool_name)
    cleaned = re.sub(pattern, "this capability", description, flags=re.IGNORECASE)
    cleaned = re.sub(r"\btool\b", "capability", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("MCP", "interface")
    return cleaned


# ---------------------------------------------------------------------------
# Dependency heuristics
# ---------------------------------------------------------------------------


@dataclass
class Dependency:
    source_tool_id: str
    field: str
    reason: str


class DependencyGraph:
    def __init__(self, tools: Sequence[ToolRecord]):
        self.tools = list(tools)
        self.by_id = {tool.tool_id: tool for tool in tools}
        self.by_lower_name = {tool.name.lower(): tool for tool in tools}
        self.token_index = self._build_token_index()
        self.index = self._build_index()

    def _build_token_index(self) -> Dict[str, List[ToolRecord]]:
        index: Dict[str, List[ToolRecord]] = {}
        for tool in self.tools:
            tokens = set(_tokenise(tool.name) + _tokenise(tool.description))
            for token in tokens:
                if not token:
                    continue
                index.setdefault(token, []).append(tool)
        return index

    def _build_index(self) -> Dict[str, List[Dependency]]:
        index: Dict[str, List[Dependency]] = {tool.tool_id: [] for tool in self.tools}
        for tool in self.tools:
            schema = tool.input_schema or {}
            required = set(schema.get("required") or [])
            properties = schema.get("properties") or {}
            for field in required:
                dep = self._guess_dependency(tool, field, properties.get(field, {}))
                if dep:
                    index[tool.tool_id].append(dep)
        return index

    def _guess_dependency(self, tool: ToolRecord, field: str, schema: Dict[str, Any]) -> Optional[Dependency]:
        field_lower = field.lower()
        base = re.sub(r"(_id|Id)$", "", field)
        candidates = []
        # 1) look for tool names containing the base keyword
        if base:
            candidates.extend(self.token_index.get(base.lower(), []))
        # 2) look for description hints in schema
        desc = schema.get("description")
        if isinstance(desc, str):
            tokens = _tokenise(desc)
            for token in tokens:
                candidates.extend(self.token_index.get(token, []))
        # 3) heuristics based on verbs
        verbs = ("list", "get", "retrieve", "fetch")
        if field_lower.endswith("id"):
            target_keyword = field_lower[:-2]
            keyword_tools = self.token_index.get(target_keyword, []) if target_keyword else []
            for other in keyword_tools:
                if any(other.name.lower().startswith(v) for v in verbs):
                    candidates.append(other)
        if not candidates:
            return None
        # deduplicate while preserving order preference
        seen = set()
        filtered: List[ToolRecord] = []
        for cand in candidates:
            if cand.tool_id == tool.tool_id:
                continue
            if cand.tool_id in seen:
                continue
            seen.add(cand.tool_id)
            filtered.append(cand)
        if not filtered:
            return None
        # pick the most descriptive candidate (longest description)
        best = max(filtered, key=lambda c: len(c.description))
        reason = f"matches required field '{field}'"
        return Dependency(source_tool_id=best.tool_id, field=field, reason=reason)

    def dependencies_for(self, tool_id: str) -> List[Dependency]:
        return self.index.get(tool_id, [])


# ---------------------------------------------------------------------------
# Schema sampling
# ---------------------------------------------------------------------------


class SchemaSampler:
    def __init__(self, rng: random.Random):
        self.rng = rng

    def sample_object(
        self,
        schema: Dict[str, Any],
        tier: str,
        fill_optional: bool = False,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        properties = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        for key in required:
            result[key] = self._sample_value(properties.get(key, {}), (key,), tier)
        optional_keys = [k for k in properties.keys() if k not in required]
        if fill_optional:
            keys_to_fill = optional_keys
        else:
            keys_to_fill = [k for k in optional_keys if self.rng.random() < 0.4]
        for key in keys_to_fill:
            result[key] = self._sample_value(properties.get(key, {}), (key,), tier)
        return result

    def _sample_value(self, schema: Dict[str, Any], path: Tuple[str, ...], tier: str) -> Any:
        if not isinstance(schema, dict):
            return self._fallback_value(path)
        if "default" in schema:
            default = schema["default"]
            if default is not None:
                return default
        if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
            return self.rng.choice(schema["enum"])

        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            non_null = [t for t in schema_type if t != "null"]
            schema_type = non_null[0] if non_null else schema_type[0]

        if schema_type == "object" or (schema_type is None and schema.get("properties")):
            fill_optional = tier in {"hard", "hard"}
            return self.sample_object(schema, tier, fill_optional=fill_optional)
        if schema_type == "array":
            items_schema = schema.get("items") or {}
            min_items = schema.get("minItems", 1)
            max_items = schema.get("maxItems")
            if isinstance(max_items, int) and max_items >= min_items:
                length = self.rng.randint(min_items, max_items or (min_items + 1))
            else:
                span = max(min_items, 1)
                length = min(3, span + self.rng.randint(0, 2))
            length = max(length, 1)
            return [self._sample_value(items_schema, path + (str(idx),), tier) for idx in range(length)]
        if schema_type == "integer":
            minimum = schema.get("minimum", 0)
            maximum = schema.get("maximum", minimum + 10)
            if isinstance(maximum, (int, float)) and isinstance(minimum, (int, float)) and maximum >= minimum:
                return int(self.rng.uniform(minimum, maximum))
            return int(self.rng.uniform(0, 20))
        if schema_type == "number":
            minimum = schema.get("minimum", 0)
            maximum = schema.get("maximum", minimum + 10.0)
            if isinstance(maximum, (int, float)) and isinstance(minimum, (int, float)) and maximum >= minimum:
                return round(self.rng.uniform(minimum, maximum), 3)
            return round(self.rng.uniform(0.0, 1.0), 3)
        if schema_type == "boolean":
            return bool(self.rng.random() < 0.5)
        if schema_type == "string" or schema_type is None:
            fmt = schema.get("format")
            if fmt == "date-time":
                return "2024-01-05T12:30:00Z"
            if fmt == "date":
                return "2024-01-05"
            if fmt == "email":
                slug = _slug(path)
                return f"{slug}@example.com"
            if fmt == "uri":
                slug = _slug(path)
                return f"https://example.com/{slug}"
            example = schema.get("examples")
            if isinstance(example, list) and example:
                return str(self.rng.choice(example))
            slug = _slug(path)
            return f"sample_{slug}"
        return self._fallback_value(path)

    def _fallback_value(self, path: Tuple[str, ...]) -> Any:
        slug = _slug(path)
        return f"sample_{slug}"


# ---------------------------------------------------------------------------
# Task synthesis core
# ---------------------------------------------------------------------------


@dataclass
class GeneratedTask:
    instruction: str
    arguments: Dict[str, Any]
    tool: ToolRecord
    tier: str
    ic: float
    ic_bucket: str
    dependencies: List[Dependency]

    def to_generic_record(self, idx: int) -> Dict[str, Any]:
        task_id = f"{self.tool.tool_id}::{self.tier}::{idx}"
        return {
            "id": task_id,
            "tool": self.tool.tool_id,
            "instruction": self.instruction,
            "input": self.arguments,
            "expect": {
                "tool": self.tool.tool_id,
                "input": self.arguments,
            },
            "meta": {
                "tier": self.tier,
                "ic_tool": self.ic,
                "ic_tool_bucket": self.ic_bucket,
                "server_id": self.tool.server_id,
                "dependencies": [dep.source_tool_id for dep in self.dependencies],
            },
        }

    def to_fuzzy_record(self, idx: int) -> Dict[str, Any]:
        task_id = f"fuzzy::{self.tool.tool_id}::{self.tier}::{idx}"
        expect = {
            "tool": self.tool.tool_id,
            "keys": list(self.arguments.keys()),
            "value_types": {k: type(v).__name__ for k, v in self.arguments.items()},
        }
        if self.dependencies:
            expect["tool_sequence"] = [dep.source_tool_id for dep in self.dependencies] + [self.tool.tool_id]
        return {
            "id": task_id,
            "instruction": self._fuzzy_instruction(),
            "expect": expect,
            "meta": {
                "tier": self.tier,
                "ic_tool": self.ic,
                "ic_tool_bucket": self.ic_bucket,
                "server_id": self.tool.server_id,
                "dependencies": [dep.source_tool_id for dep in self.dependencies],
            },
        }

    def _fuzzy_instruction(self) -> str:
        lines = []
        cleaned = _strip_tool_mentions(self.tool.description or "", self.tool.name)
        if cleaned:
            lines.append(cleaned)
        else:
            lines.append("Use the appropriate capability from the catalogue.")
        if self.dependencies:
            dep_lines = []
            for dep in self.dependencies:
                dep_tool = dep.source_tool_id.split("::")[-1].replace("_", " ")
                dep_lines.append(f"first gather the {dep.field} via the capability related to '{dep_tool}'")
            lines.append(" and ".join(dep_lines) + ", then")
        arg_desc = ", ".join(f"{k}={v}" for k, v in self.arguments.items())
        lines.append(f"complete the objective with parameters: {arg_desc}.")
        return " ".join(lines)


class TaskGenerator:
    def __init__(self, tools: Sequence[ToolRecord], rng: random.Random):
        self.tools = list(tools)
        self.rng = rng
        self.deps = DependencyGraph(self.tools)
        self.ic_results = {result.tool.tool_id: result for result in compute_all_tool_scores(self.tools)}
        self.schema_sampler = SchemaSampler(rng)

    def _bucket(self, score: float) -> str:
        if score < 4.0:
            return "low"
        if score < 9.0:
            return "medium"
        return "high"

    def _choose_tier_flags(self, tier: str) -> Dict[str, bool]:
        if tier in {"flat", "flat-easy"}:
            return {"fill_optional": False, "obfuscate": True}
        if tier == "hard":
            return {"fill_optional": True, "obfuscate": True}
        return {"fill_optional": True, "obfuscate": True}

    def synthesise_for_tool(self, tool: ToolRecord, tier: str) -> Optional[GeneratedTask]:
        schema = tool.input_schema or {}
        if not isinstance(schema, dict) or schema.get("type") not in (None, "object"):
            return None
        flags = self._choose_tier_flags(tier)
        args = self.schema_sampler.sample_object(schema, tier, fill_optional=flags["fill_optional"])
        ic = self.ic_results[tool.tool_id].score
        bucket = self._bucket(ic)
        dependencies = self.deps.dependencies_for(tool.tool_id)
        instruction = self._render_instruction(tool, args, dependencies, tier)
        return GeneratedTask(
            instruction=instruction,
            arguments=args,
            tool=tool,
            tier=tier,
            ic=ic,
            ic_bucket=bucket,
            dependencies=dependencies,
        )

    def _render_instruction(
        self,
        tool: ToolRecord,
        arguments: Dict[str, Any],
        dependencies: List[Dependency],
        tier: str,
    ) -> str:
        cleaned = _strip_tool_mentions(tool.description or "", tool.name)
        if not cleaned:
            cleaned = f"Execute the capability that resembles '{tool.name.replace('_', ' ')}'."
        lines = [cleaned]
        if dependencies:
            dep_clauses = [
                f"obtain {dep.field} using the related capability '{dep.source_tool_id.split('::')[-1]}'"
                for dep in dependencies
            ]
            lines.append("Before running this capability, " + " and ".join(dep_clauses) + ".")
        arg_lines = []
        for key, value in arguments.items():
            if isinstance(value, list):
                rendered = ", ".join(map(str, value))
                arg_lines.append(f"- {key}: [{rendered}]")
            else:
                arg_lines.append(f"- {key}: {value}")
        if arg_lines:
            lines.append("Fill it with:")
            lines.extend(arg_lines)
        if tier == "hard" and arguments:
            lines.append("Ensure all constraints from the schema are satisfied.")
        return "\n".join(lines)

    def generate(
        self,
        tiers: Sequence[str],
        limit: Optional[int] = None,
        shuffle: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        generic_records: List[Dict[str, Any]] = []
        fuzzy_records: List[Dict[str, Any]] = []
        ordered_tools = list(self.tools)
        if shuffle:
            self.rng.shuffle(ordered_tools)
        counter = 0
        for tool in ordered_tools:
            for tier in tiers:
                task = self.synthesise_for_tool(tool, tier)
                if not task:
                    continue
                generic_records.append(task.to_generic_record(counter))
                fuzzy_records.append(task.to_fuzzy_record(counter))
                counter += 1
                if limit and counter >= limit:
                    break
            if limit and counter >= limit:
                break
        return generic_records, fuzzy_records


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------


def _parse_tiers(raw: Optional[str]) -> List[str]:
    if not raw:
        return ["realistic"]
    tiers = [tier.strip().lower() for tier in raw.replace(",", " ").split() if tier.strip()]
    canonical = {"flat", "flat-easy", "realistic", "hard"}
    result = []
    for tier in tiers:
        if tier not in canonical:
            raise ValueError(f"Unknown tier '{tier}'")
        result.append(tier)
    return result


def run_cli(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate MCP benchmark tasks")
    parser.add_argument("--tools", type=pathlib.Path, required=True, help="Path to tools JSONL catalogue")
    parser.add_argument("--generic-out", type=pathlib.Path, required=True, help="Output path for generic tasks JSONL")
    parser.add_argument("--fuzzy-out", type=pathlib.Path, required=True, help="Output path for fuzzy tasks JSONL")
    parser.add_argument("--tiers", type=str, default="realistic", help="Comma-separated tiers (flat, realistic, hard)")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    parser.add_argument("--limit", type=int, help="Optional cap on number of tasks")
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    tiers = _parse_tiers(args.tiers)

    tools = load_tools(args.tools)
    if not tools:
        raise SystemExit("No valid tools parsed from the catalogue")

    generator = TaskGenerator(tools, rng)
    generic_records, fuzzy_records = generator.generate(tiers=tiers, limit=args.limit)

    args.generic_out.parent.mkdir(parents=True, exist_ok=True)
    args.fuzzy_out.parent.mkdir(parents=True, exist_ok=True)

    with args.generic_out.open("w", encoding="utf-8") as handle:
        for record in generic_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with args.fuzzy_out.open("w", encoding="utf-8") as handle:
        for record in fuzzy_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Generated {len(generic_records)} generic tasks and {len(fuzzy_records)} fuzzy tasks")


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
