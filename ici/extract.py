#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract static interface features from JSON Schemas for ICI.
Input : --tools  data/prelim/tools.jsonl
Output: --out    data/prelim/tools.features_ici.jsonl
"""
import argparse, json, re, sys
from typing import Any, Dict, List

CONSTRAINT_KEYS = {"pattern","format","minimum","maximum","minItems","maxItems",
                   "dependentRequired","dependencies"}

def _desc_tokens(txt: str) -> int:
    if not txt: return 0
    return len(re.findall(r"\w+", txt.lower()))

def _name_readability(names: List[str]) -> float:
    # penalize snake/camel/digits/very short identifiers
    if not names: return 0.0
    score = 0.0
    for n in names:
        parts = re.split(r"[_\-]|(?<=[a-z])(?=[A-Z])", n)
        if len(parts) >= 2: score += 1.0
        if re.search(r"[0-9]", n): score += 0.5
        if len(n) <= 2: score += 0.5
    return score / len(names)

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

def _count_required(s: Any) -> int:
    c = 0
    if isinstance(s, dict):
        if "required" in s and isinstance(s["required"], list): c += len(s["required"])
        if "properties" in s and isinstance(s["properties"], dict):
            for v in s["properties"].values(): c += _count_required(v)
        if s.get("type") == "array" and "items" in s:
            c += _count_required(s["items"])
        for k in ("oneOf","anyOf","allOf","if","then","else"):
            if k in s:
                v = s[k]
                if isinstance(v, list):
                    for x in v: c += _count_required(x)
                elif isinstance(v, dict):
                    c += _count_required(v)
    elif isinstance(s, list):
        for x in s: c += _count_required(x)
    return c

def _count_depth(s: Any, cur: int = 0) -> int:
    if isinstance(s, dict):
        ds = []
        if "properties" in s and isinstance(s["properties"], dict):
            for v in s["properties"].values(): ds.append(_count_depth(v, cur+1))
        if s.get("type") == "array" and "items" in s:
            ds.append(_count_depth(s["items"], cur+1))
        for k in ("oneOf","anyOf","allOf","if","then","else"):
            if k in s:
                v = s[k]
                if isinstance(v, list):
                    for x in v: ds.append(_count_depth(x, cur+1))
                elif isinstance(v, dict):
                    ds.append(_count_depth(v, cur+1))
        return max([cur] + ds) if ds else cur
    elif isinstance(s, list):
        return max((_count_depth(x, cur) for x in s), default=cur)
    return cur

def _count_branching(s: Any) -> int:
    c = 0
    if isinstance(s, dict):
        for k in ("oneOf","anyOf","allOf","if","then","else"):
            if k in s:
                v = s[k]
                c += len(v) if isinstance(v, list) else 1
        if "properties" in s and isinstance(s["properties"], dict):
            for v in s["properties"].values(): c += _count_branching(v)
        if s.get("type") == "array" and "items" in s:
            c += _count_branching(s["items"])
        for k, v in s.items():
            if isinstance(v, dict) and k not in ("properties","items"):
                c += _count_branching(v)
    elif isinstance(s, list):
        for x in s: c += _count_branching(x)
    return c

def _count_enums(s: Any):
    cnt, sizes = 0, []
    if isinstance(s, dict):
        if "enum" in s and isinstance(s["enum"], list):
            cnt += 1; sizes.append(len(s["enum"]))
        if "properties" in s and isinstance(s["properties"], dict):
            for v in s["properties"].values():
                c, ss = _count_enums(v); cnt += c; sizes += ss
        if s.get("type") == "array" and "items" in s:
            c, ss = _count_enums(s["items"]); cnt += c; sizes += ss
        for k in ("oneOf","anyOf","allOf","if","then","else"):
            if k in s:
                v = s[k]
                if isinstance(v, list):
                    for x in v: c, ss = _count_enums(x); cnt += c; sizes += ss
                elif isinstance(v, dict):
                    c, ss = _count_enums(v); cnt += c; sizes += ss
    elif isinstance(s, list):
        for x in s:
            c, ss = _count_enums(x); cnt += c; sizes += ss
    return cnt, sizes

def _count_constraints(s: Any) -> int:
    c = 0
    if isinstance(s, dict):
        for k in CONSTRAINT_KEYS:
            if k in s: c += 1
        if "properties" in s and isinstance(s["properties"], dict):
            for v in s["properties"].values(): c += _count_constraints(v)
        if s.get("type") == "array" and "items" in s:
            c += _count_constraints(s["items"])
        for k in ("oneOf","anyOf","allOf","if","then","else"):
            if k in s:
                v = s[k]
                if isinstance(v, list):
                    for x in v: c += _count_constraints(x)
                elif isinstance(v, dict):
                    c += _count_constraints(v)
    elif isinstance(s, list):
        for x in s: c += _count_constraints(x)
    return c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.tools, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    out = open(args.out, "w", encoding="utf-8")
    for t in rows:
        sch = t.get("input_schema", {})
        props = list(set(_list_props(sch)))
        num_params = len(props)
        num_required = _count_required(sch)
        depth = _count_depth(sch, 0)
        branching = _count_branching(sch)
        enum_cnt, enum_sizes = _count_enums(sch)
        enum_avg = (sum(enum_sizes)/len(enum_sizes)) if enum_sizes else 0.0
        constraints = _count_constraints(sch)
        desc_tokens = _desc_tokens(t.get("title","")) + _desc_tokens(t.get("description",""))
        name_readability = _name_readability(props)
        out.write(json.dumps({
            "tool_id": t["tool_id"],
            "num_params": num_params,
            "num_required": num_required,
            "depth": depth,
            "branching": branching,
            "enum_cnt": enum_cnt,
            "enum_avg": enum_avg,
            "constraints": constraints,
            "desc_tokens": desc_tokens,
            "name_readability": name_readability
        }, ensure_ascii=False) + "\n")
    out.close()

if __name__ == "__main__":
    main()
