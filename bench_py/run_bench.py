#!/usr/bin/env python3
"""
Benchmark runner for MCP server + agent across providers (Ollama, Gemini).

- Starts/uses the Python server on a chosen port with a given tools JSONL.
- Loads a tasks JSONL (default tests/advanced_tasks.jsonl).
  Each task should have: { id, prompt, expect?: { tool } }.
  We evaluate tool-selection accuracy and overall latency; we do not require
  the server to compute true results, only schema-valid placeholders.
- Runs the production agent using selected provider + model.
  Summarizes: success count (tool match), failures, average latency.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse
import http.client

from bench_py.agent_executor import AgentConfig, AgentExecutor

# Load .env support (lightweight fallback if python-dotenv is unavailable)
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    def load_dotenv(path: str = '.env'):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith('#'):
                        continue
                    if '=' in s:
                        k, v = s.split('=', 1)
                        k = k.strip(); v = v.strip().strip('"').strip("'")
                        os.environ.setdefault(k, v)
        except FileNotFoundError:
            pass


def wait_for_health(server_url: str, timeout_s: int = 10) -> bool:
    u = urlparse(server_url)
    conn_cls = http.client.HTTPSConnection if u.scheme == 'https' else http.client.HTTPConnection
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            conn = conn_cls(u.hostname, u.port or (443 if u.scheme == 'https' else 80), timeout=2)
            conn.request('GET', '/health')
            resp = conn.getresponse()
            ok = resp.status == 200
            resp.read()
            conn.close()
            if ok:
                return True
        except Exception:
            time.sleep(0.2)
    return False


def load_tasks(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def run_suite(server_url: str, tasks: List[Dict[str, Any]], provider: str, model: str, ollama_url: str, gemini_api_key: str, limit_tools: int, max_steps: int, retry: int) -> Dict[str, Any]:
    ok = 0
    fail = 0
    total = 0
    times: List[float] = []
    tool_matches: List[Tuple[str, str]] = []  # (expected, got)
    errors: List[str] = []

    for t in tasks:
        total += 1
        prompt = t.get('prompt') or ''
        expect = t.get('expect') or {}
        expect_tool = expect.get('tool')
        cfg = AgentConfig(
            server=server_url,
            provider=provider,
            model=model,
            ollama=ollama_url,
            gemini_api_key=gemini_api_key,
            limit_tools=limit_tools,
            max_steps=max_steps,
            retry_per_step=retry,
        )
        ex = AgentExecutor(cfg)
        start = time.time()
        try:
            res = ex.run(prompt)
            dur = time.time() - start
            times.append(dur)
            trace = res.get('trace') or []
            got_tool = trace[0]['tool'] if trace and 'tool' in trace[0] else None
            tool_matches.append((expect_tool, got_tool))
            if expect_tool and got_tool == expect_tool:
                ok += 1
            else:
                fail += 1
        except Exception as e:
            dur = time.time() - start
            times.append(dur)
            fail += 1
            errors.append(f"{t.get('id')}: {e}")

    avg = sum(times) / len(times) if times else 0.0
    p50 = sorted(times)[len(times)//2] if times else 0.0
    p95 = sorted(times)[int(len(times)*0.95)-1] if times else 0.0
    return {
        'provider': provider,
        'model': model,
        'total': total,
        'ok_tool_match': ok,
        'fail': fail,
        'avg_s': round(avg, 3),
        'p50_s': round(p50, 3),
        'p95_s': round(p95, 3),
        'errors': errors,
        'tool_matches': tool_matches,
    }


def main() -> None:
    # Load .env so GEMINI_API_KEY is visible to defaults and runtime
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=3015)
    ap.add_argument('--tools', type=str, default='bench/tools.advanced.jsonl')
    ap.add_argument('--tasks', type=str, default='tests/advanced_tasks.jsonl')
    ap.add_argument('--limit', type=int, default=20)
    ap.add_argument('--max-steps', type=int, default=4)
    ap.add_argument('--retry', type=int, default=1)
    # Providers and models
    ap.add_argument('--ollama-model', type=str, default='qwen2.5:7b-instruct')
    ap.add_argument('--ollama-url', type=str, default='http://localhost:11434')
    ap.add_argument('--gemini-model', type=str, default='gemini-2.5-flash')
    ap.add_argument('--gemini-api-key', type=str, default=None)
    ns = ap.parse_args()

    server_url = f"http://localhost:{ns.port}"
    # Start server
    proc = subprocess.Popen([
        sys.executable, 'server_py/app.py', '--port', str(ns.port), '--tools', ns.tools, '--default-latency', '0'
    ])
    try:
        if not wait_for_health(server_url, timeout_s=15):
            print('Server did not become healthy in time.', file=sys.stderr)
            proc.terminate(); proc.wait(timeout=5)
            sys.exit(1)
        tasks = load_tasks(ns.tasks)

        results: List[Dict[str, Any]] = []

        # Try Ollama if reachable
        try:
            u = urlparse(ns.ollama_url)
            conn_cls = http.client.HTTPSConnection if u.scheme == 'https' else http.client.HTTPConnection
            conn = conn_cls(u.hostname, u.port or (443 if u.scheme == 'https' else 80), timeout=2)
            conn.request('GET', '/api/tags')
            resp = conn.getresponse(); resp.read(); conn.close()
            if resp.status == 200:
                print(f"Running suite with Ollama model={ns.ollama_model}")
                results.append(run_suite(server_url, tasks, 'ollama', ns.ollama_model, ns.ollama_url, '', ns.limit, ns.max_steps, ns.retry))
            else:
                print('Ollama not reachable (tags status != 200); skipping.')
        except Exception:
            print('Ollama not reachable; skipping.')

        # Gemini
        gemini_key = ns.gemini_api_key or os.environ.get('GEMINI_API_KEY')
        if gemini_key:
            print(f"Running suite with Gemini model={ns.gemini_model}")
            results.append(run_suite(server_url, tasks, 'gemini', ns.gemini_model, ns.ollama_url, gemini_key, ns.limit, ns.max_steps, ns.retry))
        else:
            print('GEMINI_API_KEY not provided; skipping Gemini.')

        print('\n=== Bench Summary ===')
        for r in results:
            print(json.dumps(r, ensure_ascii=False))

    finally:
        try:
            proc.terminate(); proc.wait(timeout=5)
        except Exception:
            pass


if __name__ == '__main__':
    main()
