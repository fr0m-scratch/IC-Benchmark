#!/usr/bin/env python3
"""HTTP server with interface complexity controls."""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from jsonschema import ValidationError, validate

from analysis.ic_score import ToolRecord, score_tool
from server_py.ic_wrap import describe_wrapper_modes, wrap_schema


@dataclass
class ToolEntry:
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Optional[Dict[str, Any]]
    latency_ms: Optional[int]
    raw: Dict[str, Any]
    server_id: str
    ic_score: float
    schema_variants: Dict[str, Dict[str, Any]]

    def as_payload(self, wrap_mode: str) -> Dict[str, Any]:
        schema = self.schema_variants.get(wrap_mode, self.input_schema)
        payload = {
            "name": self.name,
            "description": self.description,
            "input_schema": schema,
            "latency_ms": self.latency_ms,
            "server_id": self.server_id,
            "ic_tool": self.ic_score,
        }
        if self.output_schema:
            payload["output_schema"] = self.output_schema
        payload.update({k: v for k, v in self.raw.items() if k not in payload})
        return payload


class ToolsStore:
    def __init__(self, path: str, wrap_mode: str):
        self.path = path
        self.wrap_mode = wrap_mode
        self.mtime = 0.0
        self.tools: Dict[str, ToolEntry] = {}
        self.lock = threading.RLock()

    def load(self, names_filter: Optional[List[str]] = None) -> Dict[str, ToolEntry]:
        names_filter = names_filter or []
        with self.lock:
            try:
                stat = os.stat(self.path)
            except FileNotFoundError:
                return {}
            reload_needed = stat.st_mtime != self.mtime or not self.tools
            if reload_needed:
                self.tools = self._load_all()
                self.mtime = stat.st_mtime
            if names_filter:
                return {name: entry for name, entry in self.tools.items() if name in names_filter}
            return dict(self.tools)

    def _load_all(self) -> Dict[str, ToolEntry]:
        tools: Dict[str, ToolEntry] = {}
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    entry = self._normalise_tool(raw)
                    if not entry:
                        continue
                    tools[entry.name] = entry
        except FileNotFoundError:
            return {}
        return tools

    def _normalise_tool(self, raw: Dict[str, Any]) -> Optional[ToolEntry]:
        name = raw.get("tool_name") or raw.get("name") or raw.get("id")
        if not name:
            return None
        description = raw.get("tool_description") or raw.get("description") or ""
        input_schema = raw.get("input_schema") or raw.get("inputSchema")
        if not isinstance(input_schema, dict):
            return None
        output_schema = raw.get("output_schema") if isinstance(raw.get("output_schema"), dict) else None
        latency = raw.get("latency_ms") if isinstance(raw.get("latency_ms"), int) else None
        server_id = str(raw.get("server_id") or raw.get("serverId") or raw.get("server") or "unknown")
        tool_id = raw.get("tool_id") or f"{server_id}::{name}"
        record = ToolRecord(
            tool_id=str(tool_id),
            name=str(name),
            description=str(description),
            input_schema=input_schema,
            output_schema=output_schema,
            server_id=server_id,
            server_name=server_id,
            raw=raw,
        )
        ic_result = score_tool(record)
        schema_variants = {
            mode: wrap_schema(input_schema, mode)
            for mode in describe_wrapper_modes().keys()
        }
        return ToolEntry(
            name=str(name),
            description=str(description),
            input_schema=input_schema,
            output_schema=output_schema,
            latency_ms=latency,
            raw=raw,
            server_id=server_id,
            ic_score=ic_result.score,
            schema_variants=schema_variants,
        )

    def get(self, tool_name: str) -> Optional[ToolEntry]:
        with self.lock:
            return self.tools.get(tool_name)


class RateLimiter:
    def __init__(self, max_calls: int, interval: float = 60.0):
        self.max_calls = max_calls
        self.interval = interval
        self.calls: List[float] = []
        self.lock = threading.Lock()

    def allow(self) -> bool:
        if self.max_calls <= 0:
            return True
        now = time.monotonic()
        with self.lock:
            self.calls = [ts for ts in self.calls if now - ts <= self.interval]
            if len(self.calls) >= self.max_calls:
                return False
            self.calls.append(now)
            return True


@dataclass
class ServerConfig:
    port: int
    tools_path: str
    default_latency: int
    wrap_mode: str
    validate_output: bool
    require_auth: bool
    auth_key: Optional[str]
    rate_limit: int
    paginate: bool
    page_size: int


def parse_args() -> ServerConfig:
    parser = argparse.ArgumentParser(description="MCP-like server with complexity controls")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "3001")))
    parser.add_argument("--tools", type=str, required=True)
    parser.add_argument("--default-latency", type=int, default=int(os.environ.get("DEFAULT_LATENCY", "0")))
    parser.add_argument("--wrap", choices=list(describe_wrapper_modes().keys()), default="none")
    parser.add_argument("--validate-output", action="store_true")
    parser.add_argument("--require-auth", action="store_true")
    parser.add_argument("--auth-key", type=str)
    parser.add_argument("--rate-limit", type=int, default=0, help="Maximum invoke calls per 60s (0 disables)")
    parser.add_argument("--paginate", action="store_true", help="Simulate pagination on /tools")
    parser.add_argument("--page-size", type=int, default=10)
    args = parser.parse_args()
    return ServerConfig(
        port=args.port,
        tools_path=args.tools,
        default_latency=args.default_latency,
        wrap_mode=args.wrap,
        validate_output=args.validate_output,
        require_auth=args.require_auth,
        auth_key=args.auth_key,
        rate_limit=args.rate_limit,
        paginate=args.paginate,
        page_size=args.page_size,
    )


class Handler(BaseHTTPRequestHandler):
    server_version = "mcp-ic/0.2"
    tools_store: ToolsStore
    config: ServerConfig
    rate_limiter: RateLimiter

    def _auth_ok(self) -> bool:
        if not self.config.require_auth:
            return True
        key = self.headers.get("X-API-Key") or self.headers.get("Authorization")
        expected = self.config.auth_key or "demo-key"
        return key == expected

    def _send_json(self, status: int, obj: Any, headers: Optional[Dict[str, str]] = None) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        if headers:
            for key, value in headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    # NDJSON ---------------------------------------------------------------
    def _begin_stream(self, headers: Optional[Dict[str, str]] = None) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        if headers:
            for key, value in headers.items():
                self.send_header(key, value)
        self.send_header("Connection", "keep-alive")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _write_chunk(self, payload: Dict[str, Any]) -> None:
        data = (json.dumps(payload) + "\n").encode("utf-8")
        size = f"{len(data):x}".encode("ascii")
        self.wfile.write(size + b"\r\n" + data + b"\r\n")
        self.wfile.flush()

    def _end_stream(self) -> None:
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    # GET ------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            return self._send_json(HTTPStatus.OK, {
                "ok": True,
                "tools_path": self.tools_store.path,
                "wrap_mode": self.config.wrap_mode,
            })
        if parsed.path == "/tools":
            if not self._auth_ok():
                return self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            params = parse_qs(parsed.query)
            names_param = params.get("names", [""])[0]
            names = [name.strip() for name in names_param.split(",") if name.strip()]
            all_tools = self.tools_store.load(names_filter=names)
            tools_list = list(all_tools.values())
            if self.config.paginate:
                cursor = int(params.get("cursor", ["0"])[0] or 0)
                page_size = int(params.get("limit", [self.config.page_size])[0])
                window = tools_list[cursor:cursor + page_size]
                next_cursor = cursor + len(window)
                has_more = next_cursor < len(tools_list)
                payload = {
                    "count": len(window),
                    "tools": [entry.as_payload(self.config.wrap_mode) for entry in window],
                    "next_cursor": next_cursor if has_more else None,
                }
                return self._send_json(HTTPStatus.OK, payload)
            if "limit" in params:
                try:
                    limit = int(params.get("limit", ["0"])[0])
                    if limit > 0:
                        tools_list = tools_list[:limit]
                except ValueError:
                    pass
            payloads = [entry.as_payload(self.config.wrap_mode) for entry in tools_list]
            accept = (self.headers.get("Accept") or "").lower()
            if "application/x-ndjson" in accept:
                self._begin_stream()
                for payload in payloads:
                    self._write_chunk(payload)
                self._end_stream()
                return
            return self._send_json(HTTPStatus.OK, {"count": len(payloads), "tools": payloads})
        if parsed.path == "/":
            return self._send_json(HTTPStatus.OK, {
                "ok": True,
                "endpoints": ["/health", "/tools", "/invoke?stream=1"],
                "wrap": self.config.wrap_mode,
            })
        return self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    # POST -----------------------------------------------------------------
    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/invoke":
            return self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
        if not self._auth_ok():
            return self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
        tool_name = payload.get("tool") or payload.get("tool_name")
        if not tool_name:
            return self._send_json(HTTPStatus.BAD_REQUEST, {"error": "missing_tool"})
        tools_map = self.tools_store.load()
        entry = tools_map.get(tool_name)
        if not entry:
            return self._send_json(HTTPStatus.NOT_FOUND, {"error": "tool_not_found", "tool": tool_name})
        schema = entry.schema_variants.get(self.config.wrap_mode, entry.input_schema)
        input_obj = payload.get("input") or payload.get("params") or {}
        latency_override = payload.get("latency_ms")
        latency_ms = _resolve_latency(entry.latency_ms, latency_override, self.config.default_latency)
        if not self.rate_limiter.allow():
            return self._send_json(HTTPStatus.TOO_MANY_REQUESTS, {"error": "rate_limited"})
        validation_error = _validate_input(schema, input_obj)
        headers = {"X-IC-Tool": f"{entry.ic_score:.3f}"}
        stream = _should_stream(parsed, self.headers)
        if validation_error:
            err_payload = {
                "error": {
                    "code": "invalid_input",
                    "message": validation_error,
                }
            }
            if stream:
                self._begin_stream(headers=headers)
                _emit_error_stream(self, tool_name, err_payload["error"], latency_ms)
                self._end_stream()
            else:
                self._send_json(HTTPStatus.BAD_REQUEST, err_payload, headers=headers)
            return
        response_obj = _simulate_tool(entry, input_obj, latency_ms, self.config.validate_output, schema)
        if stream:
            self._begin_stream(headers=headers)
            for event in response_obj["events"]:
                self._write_chunk(event)
            self._end_stream()
            return
        self._send_json(HTTPStatus.OK, response_obj["result"], headers=headers)


# Helpers ----------------------------------------------------------------------

def _should_stream(parsed, headers) -> bool:
    q = parse_qs(parsed.query)
    if q.get("stream", ["0"])[0] in ("1", "true"):
        return True
    return headers.get("x-stream") == "1"


def _resolve_latency(tool_latency: Optional[int], override: Any, default: int) -> int:
    if isinstance(override, int):
        return max(0, override)
    if isinstance(tool_latency, int):
        return max(0, tool_latency)
    return max(0, default)


def _validate_input(schema: Dict[str, Any], payload: Any) -> Optional[str]:
    try:
        validate(payload, schema)
    except ValidationError as exc:
        return exc.message
    return None


def _simulate_tool(
    entry: ToolEntry,
    input_obj: Dict[str, Any],
    latency_ms: int,
    validate_output: bool,
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    start_ts = int(time.time() * 1000)
    events: List[Dict[str, Any]] = []
    events.append({"event": "start", "tool": entry.name, "ts": start_ts})
    if latency_ms:
        events.append({"event": "progress", "tool": entry.name, "message": "working", "ts": start_ts + latency_ms // 2})
    output = _placeholder_from_schema(entry.output_schema, input_obj)
    if validate_output and entry.output_schema:
        err = _validate_input(entry.output_schema, output)
        if err:
            events.append({
                "event": "error",
                "tool": entry.name,
                "error": {"code": "invalid_output", "message": err},
                "ts": start_ts + latency_ms,
            })
            events.append({"event": "end", "tool": entry.name, "ts": start_ts + latency_ms})
            return {"events": events, "result": {"ok": False, "error": err}}
    events.append({
        "event": "result",
        "tool": entry.name,
        "output": output,
        "ts": start_ts + latency_ms,
    })
    events.append({"event": "end", "tool": entry.name, "ts": start_ts + latency_ms})
    return {
        "events": events,
        "result": {
            "ok": True,
            "tool": entry.name,
            "output": output,
            "latency_ms": latency_ms,
        },
    }


def _emit_error_stream(handler: Handler, tool: str, error: Dict[str, Any], latency_ms: int) -> None:
    start_ts = int(time.time() * 1000)
    handler._write_chunk({"event": "start", "tool": tool, "ts": start_ts})
    handler._write_chunk({"event": "error", "tool": tool, "error": error, "ts": start_ts + latency_ms})
    handler._write_chunk({"event": "end", "tool": tool, "ts": start_ts + latency_ms})


def _placeholder_from_schema(schema: Optional[Dict[str, Any]], input_obj: Dict[str, Any]) -> Any:
    if not schema:
        return {"echo": input_obj}
    typ = schema.get("type") if isinstance(schema, dict) else None
    if isinstance(typ, list):
        non_null = [t for t in typ if t != "null"]
        typ = non_null[0] if non_null else None
    if typ == "object" or (typ is None and isinstance(schema.get("properties"), dict)):
        result: Dict[str, Any] = {}
        for key, subschema in schema.get("properties", {}).items():
            result[key] = _placeholder_from_schema(subschema, input_obj)
        result.setdefault("echo", input_obj)
        return result
    if typ == "array":
        items = schema.get("items") if isinstance(schema.get("items"), dict) else {}
        return [_placeholder_from_schema(items, input_obj)]
    if typ == "string":
        return ""
    if typ in ("integer", "number"):
        return 0
    if typ == "boolean":
        return False
    return {"echo": input_obj}


def run_server(config: ServerConfig) -> None:
    tools_store = ToolsStore(config.tools_path, config.wrap_mode)
    tools_store.load()  # warm cache
    Handler.tools_store = tools_store
    Handler.config = config
    Handler.rate_limiter = RateLimiter(config.rate_limit)
    server = ThreadingHTTPServer(("0.0.0.0", config.port), Handler)
    print(f"[mcp-ic] listening on http://localhost:{config.port}")
    print(f"[mcp-ic] tools file: {config.tools_path} (wrap={config.wrap_mode})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[mcp-ic] shutting down")
    finally:
        server.server_close()


def main() -> None:
    config = parse_args()
    run_server(config)


if __name__ == "__main__":
    main()
