#!/usr/bin/env python3
"""
Generic offline evaluator using direct tool invocation.

Task format (JSONL), minimal required per line:
  { "id": str, "tool": str, "input": object }

Optional fields are ignored by the evaluator. No tool-specific heuristics are used.
The evaluator fetches the tool list from the server and, when available, validates
the response output against the tool's output_schema (required keys + primitive types).
"""
import json
import argparse
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
import http.client


def fetch_tools(server: str):
    u = urlparse(server)
    conn_cls = http.client.HTTPSConnection if u.scheme == 'https' else http.client.HTTPConnection
    conn = conn_cls(u.hostname, u.port or (443 if u.scheme == 'https' else 80), timeout=30)
    conn.request('GET', '/tools?limit=100', headers={'Accept': 'application/json'})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    obj = json.loads(data.decode('utf-8'))
    return obj.get('tools', [])


def invoke_tool(server: str, tool: str, input_obj: Dict[str, Any]):
    u = urlparse(server)
    conn_cls = http.client.HTTPSConnection if u.scheme == 'https' else http.client.HTTPConnection
    conn = conn_cls(u.hostname, u.port or (443 if u.scheme == 'https' else 80), timeout=30)
    payload = json.dumps({'tool': tool, 'input': input_obj}).encode('utf-8')
    conn.request('POST', '/invoke', body=payload, headers={'Content-Type': 'application/json'})
    resp = conn.getresponse()
    data = json.loads(resp.read().decode('utf-8'))
    conn.close()
    return resp.status, data


def _check_type(value: Any, expected: Union[str, List[str]]) -> bool:
    types = [expected] if isinstance(expected, str) else list(expected)
    for t in types:
        if t == 'string' and isinstance(value, str):
            return True
        if t == 'number' and isinstance(value, (int, float)):
            return True
        if t == 'integer' and isinstance(value, int):
            return True
        if t == 'boolean' and isinstance(value, bool):
            return True
        if t == 'object' and isinstance(value, dict):
            return True
        if t == 'array' and isinstance(value, list):
            return True
        if t == 'null' and value is None:
            return True
    return False


def validate_against_schema(obj: Any, schema: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(schema, dict):
        return None
    s_type = schema.get('type')
    if s_type is None:
        return None
    # object handling
    if s_type == 'object':
        if not isinstance(obj, dict):
            return 'expected_object'
        props = schema.get('properties') or {}
        required = schema.get('required') or []
        for key in required:
            if key not in obj:
                return f'missing_required:{key}'
        for k, v in obj.items():
            ps = props.get(k) if isinstance(props, dict) else None
            if isinstance(ps, dict) and 'type' in ps:
                if not _check_type(v, ps['type']):
                    return f'type_error:{k}'
        return None
    # array handling (shallow type check)
    if s_type == 'array':
        if not isinstance(obj, list):
            return 'expected_array'
        items = schema.get('items')
        if isinstance(items, dict) and 'type' in items:
            for i, it in enumerate(obj):
                if not _check_type(it, items['type']):
                    return f'item_type_error:{i}'
        return None
    # primitives
    if not _check_type(obj, s_type):
        return 'type_error'
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--server', type=str, default='http://localhost:3000')
    ap.add_argument('--tasks', type=str, default='tests/generic_tasks.jsonl')
    ns = ap.parse_args()
    server = ns.server
    tools = fetch_tools(server)
    tool_map: Dict[str, Dict[str, Any]] = {t['name']: t for t in tools}
    ok = 0
    total = 0
    with open(ns.tasks, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            t = json.loads(line)
            tool = t.get('tool')
            input_obj = t.get('input', {})
            if not tool or tool not in tool_map:
                print(f"[skip] {t.get('id')} tool {tool} not available")
                continue
            status, data = invoke_tool(server, tool, input_obj)
            success = (status == 200 and data.get('ok') is True)
            if success:
                out = data.get('output')
                schema = tool_map[tool].get('output_schema')
                err = validate_against_schema(out, schema)
                if err is None:
                    ok += 1
                    print(f"[ok] {t.get('id')}")
                else:
                    print(f"[schema mismatch] {t.get('id')} -> {err}; output={out}")
            else:
                print(f"[fail] {t.get('id')} status={status} data={data}")
    print(f"Summary: {ok}/{total} passed")


if __name__ == '__main__':
    main()
