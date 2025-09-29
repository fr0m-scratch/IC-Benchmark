#!/usr/bin/env python3
"""
Python streamable HTTP MCP-like server

Endpoints:
- GET /health -> { ok: true }
- GET /tools?limit=&names= -> list tools (JSON or NDJSON if Accept: application/x-ndjson)
- POST /invoke?stream=1 -> streams NDJSON events; else returns single JSON result
  Body: { "tool": str, "input": object, "latency_ms"?: int }

Tools JSONL format per line (flexible keys; normalized to {name, description, input_schema, output_schema?, latency_ms?}):
{
  "tool_name": "adder",
  "tool_description": "Add two numbers",
  "input_schema": {...},
  "output_schema": {...},
  "latency_ms": 0
}

Latency defaults to 0 unless overridden by request or tool definition or --default-latency.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=int(os.environ.get('PORT', '3000')))
    p.add_argument('--tools', type=str, default=os.environ.get('TOOLS_PATH', 'bench/tools.sample.jsonl'))
    p.add_argument('--default-latency', type=int, default=int(os.environ.get('DEFAULT_LATENCY', '0')))
    return p.parse_args()


def normalize_tool(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    name = obj.get('tool_name') or obj.get('name') or obj.get('id') or obj.get('tool') or obj.get('title')
    desc = obj.get('tool_description') or obj.get('description') or ''
    input_schema = obj.get('input_schema') or obj.get('inputSchema') or obj.get('schema')
    output_schema = obj.get('output_schema') or obj.get('outputSchema')
    latency_ms = obj.get('latency_ms')
    if not name or not input_schema:
        return None
    return {
        'name': name,
        'description': desc,
        'input_schema': input_schema,
        'output_schema': output_schema,
        'latency_ms': latency_ms if isinstance(latency_ms, int) else None,
    }


class ToolsStore:
    def __init__(self, path: str):
        self.path = path
        self.mtime = 0.0
        self.map: Dict[str, Dict[str, Any]] = {}

    def load(self, names_filter: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        names_filter = names_filter or []
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            return self.map
        if st.st_mtime == self.mtime and self.map:
            # cached
            if names_filter:
                return {k: v for k, v in self.map.items() if k in names_filter}
            return self.map
        # reload
        new_map: Dict[str, Dict[str, Any]] = {}
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        t = normalize_tool(obj)
                        if not t:
                            continue
                        if names_filter and t['name'] not in names_filter:
                            continue
                        if t['name'] not in new_map:
                            new_map[t['name']] = t
                    except Exception:
                        # skip malformed line
                        pass
        except Exception:
            # leave empty if read fails
            new_map = {}
        self.mtime = st.st_mtime
        self.map = new_map
        return self.map


def placeholder_from_schema(schema: Any, input_obj: Any) -> Any:
    if not isinstance(schema, dict):
        return {'echo': input_obj}
    t = schema.get('type')
    # handle union types like ["string","null"] by preferring null, else first
    if isinstance(t, list):
        if 'null' in t:
            return None
        t = t[0] if t else None
    if t == 'object':
        out: Dict[str, Any] = {}
        props = schema.get('properties') or {}
        if isinstance(props, dict):
            for k, v in props.items():
                out[k] = placeholder_from_schema(v, None)
        out['echo'] = input_obj
        return out
    if t == 'array':
        return []
    if t == 'string':
        return ''
    if t in ('number', 'integer'):
        return 0
    if t == 'boolean':
        return False
    return {'echo': input_obj}


def validate_input(input_obj: Any, schema: Any) -> Optional[str]:
    """Very basic validation: required fields and primitive type checks."""
    if not isinstance(schema, dict):
        return None
    required = schema.get('required') or []
    props = schema.get('properties') or {}
    if not isinstance(input_obj, dict):
        return 'input_must_be_object'
    for key in required:
        if key not in input_obj:
            return f'missing_required:{key}'
    # primitive type checks
    for k, v in input_obj.items():
        ps = props.get(k) if isinstance(props, dict) else None
        if ps and 'type' in ps:
            t = ps['type']
            if t == 'string' and not isinstance(v, str):
                return f'type_error:{k}:expected_string'
            if t in ('number', 'integer') and not isinstance(v, (int, float)):
                return f'type_error:{k}:expected_number'
            if t == 'boolean' and not isinstance(v, bool):
                return f'type_error:{k}:expected_boolean'
            if t == 'object' and not isinstance(v, dict):
                return f'type_error:{k}:expected_object'
            if t == 'array' and not isinstance(v, list):
                return f'type_error:{k}:expected_array'
    return None


def simulate_events(tool: str, input_obj: Any, tool_def: Dict[str, Any], latency_ms: int) -> List[Dict[str, Any]]:
    now = int(time.time() * 1000)
    events: List[Dict[str, Any]] = []
    events.append({'event': 'start', 'tool': tool, 'ts': now})
    total = max(0, int(latency_ms))
    if total > 0:
        events.append({'event': 'progress', 'tool': tool, 'message': 'working', 'ts': now + total // 2})
    # Validate input
    err = validate_input(input_obj, tool_def.get('input_schema'))
    if err:
        events.append({'event': 'error', 'tool': tool, 'error': {'code': 'invalid_input', 'message': err}, 'ts': now + total})
        events.append({'event': 'end', 'tool': tool, 'ts': now + total})
        return events

    # Schema-derived placeholder output
    output_schema = tool_def.get('output_schema')
    output = placeholder_from_schema(output_schema, input_obj)
    events.append({'event': 'result', 'tool': tool, 'output': output, 'ts': now + total})
    events.append({'event': 'end', 'tool': tool, 'ts': now + total})
    return events


class Handler(BaseHTTPRequestHandler):
    server_version = 'mcp-py/0.1'
    tools_store: ToolsStore = None  # type: ignore
    default_latency: int = 0

    def _send_json(self, status: int, obj: Any) -> None:
        data = json.dumps(obj).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _begin_ndjson(self) -> None:
        self.send_response(200)
        self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Transfer-Encoding', 'chunked')
        self.end_headers()

    def _write_chunk(self, data: bytes) -> None:
        # HTTP/1.1 chunked transfer
        size = ('%x' % len(data)).encode('ascii')
        self.wfile.write(size + b"\r\n" + data + b"\r\n")
        self.wfile.flush()

    def _end_chunks(self) -> None:
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def do_GET(self) -> None:  # noqa: N802 (method name by BaseHTTPRequestHandler)
        parsed = urlparse(self.path)
        if parsed.path == '/health':
            return self._send_json(200, {'ok': True, 'tools_path': self.tools_store.path})
        if parsed.path == '/tools':
            q = parse_qs(parsed.query)
            names = []
            if 'names' in q:
                names = [s.strip() for s in ','.join(q.get('names', [])).split(',') if s.strip()]
            tools_map = self.tools_store.load(names_filter=names)
            tools = list(tools_map.values())
            if 'limit' in q:
                try:
                    limit = int(q['limit'][0])
                    tools = tools[:max(0, limit)]
                except Exception:
                    pass
            accept = (self.headers.get('Accept') or '').lower()
            if 'application/x-ndjson' in accept:
                self._begin_ndjson()
                for t in tools:
                    self._write_chunk((json.dumps(t) + '\n').encode('utf-8'))
                self._end_chunks()
                return
            return self._send_json(200, {'count': len(tools), 'tools': tools})
        if parsed.path == '/':
            return self._send_json(200, {
                'ok': True,
                'endpoints': ['/health', '/tools', '/invoke?stream=1'],
                'tools_path': self.tools_store.path,
            })
        self._send_json(404, {'error': 'not_found'})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != '/invoke':
            return self._send_json(404, {'error': 'not_found'})
        length = int(self.headers.get('Content-Length') or '0')
        raw = self.rfile.read(length) if length > 0 else b'{}'
        try:
            payload = json.loads(raw.decode('utf-8') or '{}')
        except Exception:
            return self._send_json(400, {'error': 'invalid_json'})

        q = parse_qs(parsed.query)
        do_stream = (q.get('stream', ['0'])[0] in ('1', 'true')) or (self.headers.get('x-stream') == '1')
        tool_name = payload.get('tool') or payload.get('tool_name')
        input_obj = payload.get('input') or payload.get('params') or {}
        override_latency = payload.get('latency_ms')

        tools_map = self.tools_store.load()
        tool_def = tools_map.get(tool_name)
        if not tool_def:
            return self._send_json(404, {'error': 'tool_not_found', 'tool': tool_name})
        latency_ms = int(override_latency) if isinstance(override_latency, int) else (
            tool_def.get('latency_ms') if isinstance(tool_def.get('latency_ms'), int) else self.default_latency
        )

        if do_stream:
            events = simulate_events(tool_name, input_obj, tool_def, latency_ms)
            self._begin_ndjson()
            # stream events with delays
            for i, ev in enumerate(events):
                if i == 1 and latency_ms > 0:  # progress after half
                    time.sleep(latency_ms / 1000.0 / 2.0)
                elif i == len(events) - 1 and latency_ms > 0:
                    # end immediately after result
                    pass
                elif i == 2 and latency_ms > 0:
                    time.sleep(latency_ms / 1000.0 / 2.0)
                self._write_chunk((json.dumps(ev) + '\n').encode('utf-8'))
            self._end_chunks()
            return

        # non-stream response
        events = simulate_events(tool_name, input_obj, tool_def, latency_ms)
        err_ev = next((e for e in events if e.get('event') == 'error'), None)
        if err_ev:
            return self._send_json(400, {'ok': False, 'error': err_ev.get('error'), 'tool': tool_name})
        result = next((e for e in events if e.get('event') == 'result'), None)
        output = result.get('output') if result else {'echo': input_obj}
        return self._send_json(200, {'ok': True, 'tool': tool_name, 'output': output, 'latency_ms': latency_ms})


def run_server(port: int, tools_path: str, default_latency: int) -> None:
    Handler.tools_store = ToolsStore(tools_path)
    Handler.default_latency = default_latency
    srv = ThreadingHTTPServer(('0.0.0.0', port), Handler)
    print(f"[mcp-py] listening on http://localhost:{port}")
    print(f"[mcp-py] tools from {tools_path}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == '__main__':
    ns = parse_args()
    run_server(ns.port, ns.tools, ns.default_latency)
