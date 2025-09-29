#!/usr/bin/env python3
"""
End-to-end runner for the Python MCP server and production agent.

Features:
- Starts the Python server on a chosen port with a given tools JSONL.
- Waits for /health.
- Runs offline evaluation on a chosen tasks JSONL via direct tool calls.
- Runs the production agent per task using a real Ollama model and checks final_answer.
"""

import argparse
import json
import subprocess
import sys
import time
from typing import Any, Dict, List
from urllib.parse import urlparse
import http.client

from bench_py.agent_executor import AgentConfig, AgentExecutor


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


def invoke_tool(server: str, tool: str, input_obj: Dict[str, Any]):
    u = urlparse(server)
    conn_cls = http.client.HTTPSConnection if u.scheme == 'https' else http.client.HTTPConnection
    conn = conn_cls(u.hostname, u.port or (443 if u.scheme == 'https' else 80), timeout=10)
    payload = json.dumps({'tool': tool, 'input': input_obj}).encode('utf-8')
    conn.request('POST', '/invoke', body=payload, headers={'Content-Type': 'application/json'})
    resp = conn.getresponse()
    data = json.loads(resp.read().decode('utf-8'))
    conn.close()
    return resp.status, data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=3001)
    ap.add_argument('--tools', type=str, default='bench/tools.advanced.jsonl')
    ap.add_argument('--tasks', type=str, default='tests/advanced_tasks.jsonl')
    ap.add_argument('--model', type=str, required=True)
    ap.add_argument('--ollama', type=str, default='http://localhost:11434')
    ap.add_argument('--max-steps', type=int, default=4)
    ap.add_argument('--retry', type=int, default=1)
    ns = ap.parse_args()

    server_url = f"http://localhost:{ns.port}"

    # Start server
    proc = subprocess.Popen([
        sys.executable, 'server_py/app.py', '--port', str(ns.port), '--tools', ns.tools
    ])
    try:
        if not wait_for_health(server_url, timeout_s=15):
            print('Server did not become healthy in time.', file=sys.stderr)
            proc.terminate()
            proc.wait(timeout=5)
            sys.exit(1)

        print(f"Server ready at {server_url}. Tools: {ns.tools}")

        # Offline eval via direct calls
        tasks = load_tasks(ns.tasks)
        offline_ok = 0
        for t in tasks:
            exp = t.get('expect') or {}
            tool = exp.get('tool')
            out_expect = exp.get('output') or {}
            # derive a simple input for some tools
            prompt = t.get('prompt', '')
            input_obj: Dict[str, Any] = {}
            if tool == 'subtractor':
                import re
                m = re.search(r"[Ss]ubtract\s+(\d+)\s+from\s+(\d+)", prompt)
                if m:
                    input_obj = {'a': int(m.group(2)), 'b': int(m.group(1))}
                else:
                    nums = list(map(int, re.findall(r"(\d+)", prompt)))
                    if len(nums) >= 2:
                        input_obj = {'a': nums[0], 'b': nums[1]}
            elif tool in ('adder', 'multiplier', 'divider'):
                import re
                nums = list(map(int, re.findall(r"(\d+)", prompt)))
                if len(nums) >= 2:
                    input_obj = {'a': nums[0], 'b': nums[1]}
            elif tool == 'power':
                import re
                nums = list(map(int, re.findall(r"(\d+)", prompt)))
                if len(nums) >= 2:
                    input_obj = {'base': nums[0], 'exponent': nums[1]}
            elif tool == 'string_upper':
                import re
                m = re.search(r"'([^']+)'", prompt)
                input_obj = {'text': m.group(1) if m else prompt}
            elif tool == 'string_concat':
                import re
                parts = re.findall(r"'([^']+)'", prompt)
                if len(parts) >= 2:
                    input_obj = {'a': parts[0], 'b': parts[1]}
            elif tool == 'string_substr':
                import re
                parts = re.findall(r"'([^']+)'", prompt)
                nums = list(map(int, re.findall(r"(\d+)", prompt)))
                text = parts[0] if parts else ''
                start = nums[0] if nums else 0
                length = nums[1] if len(nums) > 1 else None
                input_obj = {'text': text, 'start': start}
                if length is not None:
                    input_obj['length'] = length
            elif tool == 'date_add_days':
                import re
                m = re.search(r"(\d{4}-\d{2}-\d{2})", prompt)
                nums = list(map(int, re.findall(r"(\d+)", prompt)))
                date = m.group(1) if m else '2025-01-01'
                days = nums[0] if nums else 0
                input_obj = {'date': date, 'days': days}
            elif tool == 'kb_lookup':
                input_obj = {'key': 'france'}
            elif tool == 'sleep_echo':
                import re
                words = re.findall(r"([A-Za-z]+)\.?$", prompt)
                msg = words[-1] if words else 'ok'
                input_obj = {'msg': msg}

            status, data = invoke_tool(server_url, tool, input_obj)
            if status == 200 and data.get('ok'):
                out = data.get('output') or {}
                subset_ok = all(k in out and (out_expect[k] == out[k]) for k in out_expect)
                if subset_ok:
                    offline_ok += 1
                else:
                    print(f"[offline mismatch] {t['id']} expected {out_expect}, got {out}")
            else:
                print(f"[offline fail] {t['id']} status={status} data={data}")
        print(f"Offline eval: {offline_ok}/{len(tasks)} passed")

        # Online agent per task
        def numeric_equal(a: str, b: str) -> bool:
            try:
                return abs(float(a) - float(b)) < 1e-9
            except Exception:
                return False

        online_ok = 0
        for t in tasks:
            cfg = AgentConfig(server=server_url, ollama=ns.ollama, model=ns.model, limit_tools=20, max_steps=ns.max_steps, retry_per_step=ns.retry)
            ex = AgentExecutor(cfg)
            print(f"\n=== Running task {t['id']} ===")
            res = ex.run(t['prompt'])
            final = (res.get('final_answer') or '').strip()
            expect_final = str(t.get('expect_final') or '').strip()
            if expect_final and final:
                if final == expect_final or numeric_equal(final, expect_final):
                    online_ok += 1
                else:
                    print(f"[agent mismatch] {t['id']} expected '{expect_final}', got '{final}'")
            else:
                print(f"[agent info] {t['id']} final='{final}' last_tool_output={res.get('last_tool_output')}")
        print(f"Online agent (final answer match): {online_ok}/{len(tasks)}")

    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            pass


if __name__ == '__main__':
    main()
