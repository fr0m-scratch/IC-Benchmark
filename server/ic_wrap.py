"""Schema wrapping utilities for interface complexity manipulation.

The wrappers expose three modes:
- ``none``: returns the original schema untouched.
- ``flat``: produces a "flat-easy" variant that keeps only required fields,
  injects defaults/examples, and removes nested optional structure.
- ``hard``: produces a "hard" variant that tightens constraints by marking
  optional fields as required and adding synthetic guards (``allOf`` /
  ``minLength``) to increase interface burden.

These transformations are intentionally conservative so that generated schemas
remain valid JSON Schema drafts used by the server.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Tuple


def wrap_schema(schema: Dict[str, Any], mode: str) -> Dict[str, Any]:
    if mode == "none":
        return copy.deepcopy(schema)
    if mode == "flat":
        return _flat_wrap(schema)
    if mode == "hard":
        return _hard_wrap(schema)
    raise ValueError(f"Unknown wrap mode '{mode}'")


def _flat_wrap(schema: Dict[str, Any]) -> Dict[str, Any]:
    schema = copy.deepcopy(schema)
    if not isinstance(schema, dict):
        return {}
    if schema.get("type") not in (None, "object"):
        return schema
    props = schema.get("properties") or {}
    required = schema.get("required") or []
    flat_props: Dict[str, Any] = {}
    for key in required:
        subschema = props.get(key, {"type": "string"})
        flat_props[key] = _ensure_examples(subschema, path=(key,))
    schema["properties"] = flat_props
    schema["required"] = list(flat_props.keys())
    schema["description"] = schema.get("description", "") + " (flat-easy variant)"
    schema.setdefault("additionalProperties", False)
    return schema


def _hard_wrap(schema: Dict[str, Any]) -> Dict[str, Any]:
    base = copy.deepcopy(schema)
    if not isinstance(base, dict):
        return {}
    if base.get("type") not in (None, "object"):
        return base
    props = base.get("properties") or {}
    hard_props: Dict[str, Any] = {}
    required = set(base.get("required") or [])
    for key, subschema in props.items():
        hard_props[key] = _tighten_schema(subschema, path=(key,))
        required.add(key)
    base["properties"] = hard_props
    base["required"] = sorted(required)
    base.setdefault("allOf", []).append({
        "type": "object",
        "properties": {
            key: {"$ref": f"#/properties/{key}"} for key in hard_props
        },
        "additionalProperties": False,
    })
    base["description"] = base.get("description", "") + " (hard variant)"
    return base


def _ensure_examples(schema: Dict[str, Any], path: Tuple[str, ...]) -> Dict[str, Any]:
    schema = copy.deepcopy(schema)
    if not isinstance(schema, dict):
        schema = {"type": "string"}
    schema.setdefault("examples", [_sample_value(schema, path)])
    schema.setdefault("description", schema.get("description", "") + " (flattened)")
    return schema


def _tighten_schema(schema: Dict[str, Any], path: Tuple[str, ...]) -> Dict[str, Any]:
    schema = copy.deepcopy(schema)
    typ = schema.get("type")
    if isinstance(typ, list):
        typ = [t for t in typ if t != "null"] or typ
        schema["type"] = typ
    if typ == "string" or (not typ and schema.get("enum")):
        schema.setdefault("minLength", 3)
        schema.setdefault("pattern", r".+")
    elif typ == "integer":
        schema.setdefault("minimum", 1)
    elif typ == "number":
        schema.setdefault("minimum", 0.0)
    elif typ == "array":
        schema.setdefault("minItems", max(1, schema.get("minItems", 0)))
        schema.setdefault("uniqueItems", False)
        items = schema.get("items") or {"type": "string"}
        schema["items"] = _tighten_schema(items, path + ("items",))
    elif typ == "object" or (typ is None and schema.get("properties")):
        props = schema.get("properties") or {}
        tightened = {k: _tighten_schema(v, path + (k,)) for k, v in props.items()}
        schema["properties"] = tightened
        schema["required"] = sorted(set(schema.get("required") or []) | set(props.keys()))
        schema.setdefault("additionalProperties", False)
    schema.setdefault("description", schema.get("description", "") + " (tightened)")
    return schema


def _sample_value(schema: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    typ = schema.get("type")
    if isinstance(typ, list):
        typ = [t for t in typ if t != "null"] or typ
        typ = typ[0] if isinstance(typ, list) else typ
    if typ == "integer":
        return schema.get("minimum", 0)
    if typ == "number":
        return schema.get("minimum", 0.0)
    if typ == "boolean":
        return True
    if typ == "array":
        items = schema.get("items") or {"type": "string"}
        return [_sample_value(items, path + ("0",))]
    if typ == "object":
        props = schema.get("properties") or {}
        return {key: _sample_value(sub, path + (key,)) for key, sub in props.items()}
    if schema.get("enum"):
        return schema["enum"][0]
    slug = "_".join(path) if path else "value"
    return f"sample_{slug}"


def describe_wrapper_modes() -> Dict[str, str]:
    return {
        "none": "Original schema",
        "flat": "Keep required fields only, inject defaults/examples",
        "hard": "Mark optional fields as required and add extra constraints",
    }


__all__ = ["wrap_schema", "describe_wrapper_modes"]
