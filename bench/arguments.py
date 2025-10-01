"""Utilities for argument filling and schema validation."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from jsonschema import ValidationError, validate


def describe_schema(schema: Dict[str, Any]) -> str:
    """Render a compact textual description of the JSON schema."""
    if not isinstance(schema, dict):
        return "Schema unavailable."
    title = schema.get("title") or "Tool arguments"
    lines = [title]
    required = set(schema.get("required") or [])
    props = schema.get("properties") or {}
    for key, value in props.items():
        if not isinstance(value, dict):
            continue
        typ = value.get("type", "any")
        desc = value.get("description", "")
        enum = value.get("enum")
        extras = []
        if enum:
            extras.append(f"enum={len(enum)} options")
        if key in required:
            extras.append("required")
        line = f"- {key} ({typ})"
        if extras:
            line += " [" + ", ".join(extras) + "]"
        if desc:
            line += f": {desc}"
        lines.append(line)
    return "\n".join(lines)


def should_use_slot_mode(ic_score: float, threshold: float) -> bool:
    return ic_score >= threshold


def validate_arguments(schema: Dict[str, Any], arguments: Dict[str, Any]) -> Tuple[bool, str]:
    try:
        validate(arguments, schema)
        return True, ""
    except ValidationError as exc:
        return False, exc.message


def find_missing_required(schema: Dict[str, Any], arguments: Dict[str, Any]) -> List[str]:
    required = schema.get("required") or []
    missing = [key for key in required if key not in arguments]
    return missing


def slot_fill_prompt(schema: Dict[str, Any], current_args: Dict[str, Any]) -> str:
    description = describe_schema(schema)
    payload = json.dumps(current_args, ensure_ascii=False, indent=2)
    missing = find_missing_required(schema, current_args)
    missing_desc = ", ".join(missing) if missing else "(all required filled)"
    return (
        f"Schema:\n{description}\n\nCurrent arguments:\n{payload}\n\n"
        f"Missing required fields: {missing_desc}."
    )


def full_argument_prompt(schema: Dict[str, Any], task_context: str) -> str:
    description = describe_schema(schema)
    return f"Task: {task_context}\n\nSchema:\n{description}\n\nReturn STRICT JSON matching the schema."


__all__ = [
    "describe_schema",
    "should_use_slot_mode",
    "validate_arguments",
    "find_missing_required",
    "slot_fill_prompt",
    "full_argument_prompt",
    "synthesise_minimal_arguments",
]


def _default_for_schema(schema: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    if not isinstance(schema, dict):
        return f"value_{'_'.join(path) if path else 'x'}"
    if "default" in schema:
        return schema["default"]
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        non_null = [t for t in schema_type if t != "null"]
        schema_type = non_null[0] if non_null else schema_type[0]
    if schema_type == "object" or (schema_type is None and schema.get("properties")):
        return {}
    if schema_type == "array":
        return []
    if schema_type == "integer":
        return int(schema.get("minimum", 0))
    if schema_type == "number":
        return float(schema.get("minimum", 0.0))
    if schema_type == "boolean":
        return False
    return f"value_{'_'.join(path) if path else 'x'}"


def synthesise_minimal_arguments(schema: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(schema, dict):
        return {}
    if schema.get("type") not in (None, "object"):
        return {}
    props = schema.get("properties") or {}
    required = schema.get("required") or []
    result: Dict[str, Any] = {}
    for key in required:
        subschema = props.get(key, {})
        if isinstance(subschema, dict) and (subschema.get("type") in (None, "object")):
            result[key] = synthesise_minimal_arguments(subschema)
        else:
            result[key] = _default_for_schema(subschema, (key,))
    return result
