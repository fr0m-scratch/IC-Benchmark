#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flatten a tool JSON into a retrievable text doc: title + desc + param_names + enums.
Can be imported as a module by run_ir.py.
"""
import re
from typing import Any, Dict, List

def _list_props(s: Any) -> List[str]:
    names = []
    if isinstance(s, dict):
        if "properties" in s and isinstance(s["properties"], dict):
            names.extend(s["properties"].keys())
            for v in s["properties"].values():
                names.extend(_list_props(v))
        if s.get("type") == "array" and "items" in s:
            names.extend(_list_props(s["items"]))
        for k in ("oneOf","anyOf","allOf","if","then","else"):
            if k in s:
                v = s[k]
                if isinstance(v, list):
                    for x in v: names.extend(_list_props(x))
                elif isinstance(v, dict):
                    names.extend(_list_props(v))
    elif isinstance(s, list):
        for x in s: names.extend(_list_props(x))
    return names

def _collect_enums(s: Any) -> List[str]:
    toks = []
    if isinstance(s, dict):
        if "enum" in s and isinstance(s["enum"], list):
            toks += [str(x) for x in s["enum"]]
        if "properties" in s and isinstance(s["properties"], dict):
            for v in s["properties"].values():
                toks += _collect_enums(v)
        if s.get("type") == "array" and "items" in s:
            toks += _collect_enums(s["items"])
        for k in ("oneOf","anyOf","allOf","if","then","else"):
            if k in s:
                v = s[k]
                if isinstance(v, list):
                    for x in v: toks += _collect_enums(x)
                elif isinstance(v, dict):
                    toks += _collect_enums(v)
    elif isinstance(s, list):
        for x in s: toks += _collect_enums(x)
    return toks

def tool_to_text_doc(tool: Dict[str, Any]) -> str:
    sch = tool.get("input_schema", {})
    param_names = list(set(_list_props(sch)))
    enums = _collect_enums(sch)
    parts = [
        tool.get("title",""),
        tool.get("description",""),
        "param_names: " + " ".join(param_names),
        "enums: " + " ".join(enums)
    ]
    return " ".join(parts).lower()
