#!/usr/bin/env python3
"""
Production-style agent executor for MCP server + Ollama.

Capabilities:
- Planning step (LLM): derive brief plan from user prompt and tools.
- Tool selection (LLM): propose exactly one TOOL_CALL {...}.
- Streaming tool execution with error handling.
- Fallback on failure: feed error back to LLM and retry selection.
- Finalization: if no tool is needed, or after a tool, ask LLM for FINAL_ANSWER.

Conventions for LLM IO:
- Tool call line: `TOOL_CALL {"tool":"<name>", "input": {...}}`
- Final answer line: starts with `FINAL_ANSWER:` followed by free text
"""

import json
import http.client
import sys
from dataclasses import dataclass
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse, urlencode


@dataclass
class AgentConfig:
    server: str
    # LLM provider selection: 'ollama' or 'gemini'
    provider: str = 'ollama'
    # Common
    model: Optional[str] = None
    # Ollama config
    ollama: str = 'http://localhost:11434'
    # Gemini config
    gemini_api_key: Optional[str] = None
    limit_tools: int = 10
    max_steps: int = 4
    retry_per_step: int = 2


def http_json(method: str, target: str, body: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None):
    u = urlparse(target)
    conn_cls = http.client.HTTPSConnection if u.scheme == 'https' else http.client.HTTPConnection
    port = u.port or (443 if u.scheme == 'https' else 80)
    conn = conn_cls(u.hostname, port, timeout=60)
    try:
        payload = None
        hdrs = headers.copy() if headers else {}
        if body is not None:
            payload = json.dumps(body).encode('utf-8')
            hdrs['Content-Type'] = 'application/json'
            hdrs['Content-Length'] = str(len(payload))
        conn.request(method, u.path + (f"?{u.query}" if u.query else ''), body=payload, headers=hdrs)
        resp = conn.getresponse()
        data = resp.read()
        return resp.status, dict(resp.getheaders()), data
    finally:
        conn.close()


def fetch_tools(server: str, limit: int) -> List[Dict[str, Any]]:
    u = urlparse(server)
    path = f"/tools?{urlencode({'limit': limit})}"
    conn_cls = http.client.HTTPSConnection if u.scheme == 'https' else http.client.HTTPConnection
    port = u.port or (443 if u.scheme == 'https' else 80)
    conn = conn_cls(u.hostname, port, timeout=30)
    try:
        conn.request('GET', path, headers={'Accept': 'application/json'})
        resp = conn.getresponse()
        if resp.status != 200:
            raise RuntimeError(f"Failed to fetch tools: {resp.status}")
        data = resp.read()
        obj = json.loads(data.decode('utf-8'))
        return obj.get('tools', [])
    finally:
        conn.close()


def ollama_chat_stream(ollama: str, model: str, messages: List[Dict[str, str]]) -> Iterable[Dict[str, Any]]:
    u = urlparse(ollama)
    path = '/api/chat'
    conn_cls = http.client.HTTPSConnection if u.scheme == 'https' else http.client.HTTPConnection
    port = u.port or (443 if u.scheme == 'https' else 80)
    conn = conn_cls(u.hostname, port, timeout=300)
    payload = json.dumps({'model': model, 'messages': messages, 'stream': True}).encode('utf-8')
    headers = {'Content-Type': 'application/json', 'Content-Length': str(len(payload))}
    conn.request('POST', path, body=payload, headers=headers)
    resp = conn.getresponse()
    buf = b''
    while True:
        chunk = resp.read(1024)
        if not chunk:
            break
        buf += chunk
        while b'\n' in buf:
            line, buf = buf.split(b'\n', 1)
            s = line.decode('utf-8', errors='ignore').strip()
            if not s:
                continue
            try:
                ev = json.loads(s)
                yield ev
            except Exception:
                continue
    conn.close()


def _split_messages_for_gemini(messages: List[Dict[str, str]]):
    sys_parts: List[str] = []
    other_parts: List[str] = []
    for m in messages:
        role = (m.get('role') or 'user').lower()
        content = m.get('content') or ''
        if not content:
            continue
        if role == 'system':
            sys_parts.append(content)
        else:
            prefix = 'Tool' if role == 'tool' else 'User'
            other_parts.append(f"{prefix}:\n{content}")
    sys_text = "\n\n".join(sys_parts).strip()
    user_text = "\n\n".join(other_parts).strip()
    return sys_text, user_text


def gemini_chat_stream(api_key: str, model: str, messages: List[Dict[str, str]]) -> Iterable[Dict[str, Any]]:
    # Non-streaming: call generateContent once and yield a single event
    import urllib.request
    import urllib.parse
    sys_text, user_text = _split_messages_for_gemini(messages)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{urllib.parse.quote(model, safe='')}:generateContent"
    payload: Dict[str, Any] = {
        'contents': [
            {
                'role': 'user',
                'parts': [{'text': user_text or 'Respond to the user.'}],
            }
        ],
        'generationConfig': {
            'temperature': 0.2,
        },
    }
    if sys_text:
        payload['systemInstruction'] = {
            'parts': [{'text': sys_text}]
        }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            'Content-Type': 'application/json',
            'x-goog-api-key': api_key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            obj = json.loads(raw.decode('utf-8'))
    except Exception as e:
        # surface an error-like event to keep the interface consistent
        yield {'message': {'content': f"FINAL_ANSWER: Error contacting Gemini: {e}"}}
        return
    text = ''
    try:
        cands = obj.get('candidates') or []
        if cands:
            content = cands[0].get('content') or {}
            parts = content.get('parts') or []
            for p in parts:
                if 'text' in p:
                    text += str(p['text'])
    except Exception:
        pass
    if not text:
        text = json.dumps(obj)
    yield {'message': {'content': text}}


def detect_tool_call_and_final(events: Iterable[Dict[str, Any]]):
    """Return ('tool', obj) if a TOOL_CALL is detected, or ('final', text) if FINAL_ANSWER appears, else ('none', text)."""
    buf = ''
    collecting = False
    json_buf = ''
    depth = 0
    for ev in events:
        msg = ev.get('message') or {}
        delta = msg.get('content') or ''
        if not delta:
            continue
        # live print
        sys.stdout.write(delta)
        sys.stdout.flush()
        buf += delta

        # detect FINAL_ANSWER
        fa_idx = buf.rfind('FINAL_ANSWER:')
        if fa_idx >= 0:
            final_text = buf[fa_idx + len('FINAL_ANSWER:'):].strip()
            return 'final', final_text

        # detect TOOL_CALL JSON
        if not collecting:
            idx = buf.find('TOOL_CALL')
            if idx >= 0:
                tail = buf[idx + len('TOOL_CALL'):]
                br = tail.find('{')
                if br >= 0:
                    collecting = True
                    json_buf = tail[br:]
                    depth = 0
                    for ch in json_buf:
                        if ch == '{':
                            depth += 1
                        elif ch == '}':
                            depth -= 1
                    if depth == 0:
                        s = json_buf.strip()
                        if s.startswith('{') and s.endswith('}'):
                            try:
                                return 'tool', json.loads(s)
                            except Exception:
                                pass
        else:
            json_buf += delta
            for ch in delta:
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
            if depth <= 0:
                s = json_buf.strip()
                last = s.rfind('}')
                if last >= 0:
                    s = s[:last + 1]
                if s.startswith('{') and s.endswith('}'):
                    try:
                        return 'tool', json.loads(s)
                    except Exception:
                        pass
                collecting = False
                json_buf = ''
                depth = 0
    return 'none', buf


def invoke_tool_stream(server: str, tool: str, input_obj: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    u = urlparse(server)
    path = '/invoke?stream=1'
    conn_cls = http.client.HTTPSConnection if u.scheme == 'https' else http.client.HTTPConnection
    port = u.port or (443 if u.scheme == 'https' else 80)
    conn = conn_cls(u.hostname, port, timeout=300)
    payload = json.dumps({'tool': tool, 'input': input_obj}).encode('utf-8')
    headers = {'Content-Type': 'application/json', 'x-stream': '1', 'Content-Length': str(len(payload))}
    conn.request('POST', path, body=payload, headers=headers)
    resp = conn.getresponse()
    final_output: Optional[Dict[str, Any]] = None
    final_error: Optional[Dict[str, Any]] = None
    buf = b''
    print('\n')
    while True:
        chunk = resp.read(1024)
        if not chunk:
            break
        buf += chunk
        while b'\n' in buf:
            line, buf = buf.split(b'\n', 1)
            s = line.decode('utf-8', errors='ignore').strip()
            if not s:
                continue
            try:
                ev = json.loads(s)
                print(f"[tool:{tool}] {s}")
                if ev.get('event') == 'result':
                    final_output = ev.get('output')
                if ev.get('event') == 'error':
                    final_error = ev.get('error') or {'message': 'tool_error'}
            except Exception:
                continue
    conn.close()
    return final_output, final_error


class AgentExecutor:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.tools: List[Dict[str, Any]] = []

    def load_tools(self):
        self.tools = fetch_tools(self.cfg.server, self.cfg.limit_tools)

    def build_tools_context(self) -> str:
        lines = []
        lines.append('You have access to these tools:')
        for t in self.tools:
            lines.append(f"- {t.get('name')}: {t.get('description','')}")
            if t.get('input_schema'):
                lines.append(f"  input_schema: {json.dumps(t['input_schema'])}")
        return '\n'.join(lines)

    def make_prompt_plan(self, user_prompt: str) -> str:
        return (
            "Plan your approach in 1-3 concise steps for the user's task. "
            "Use the provided tools when they match the request and be precise with their input schema. "
            "After any tool call, respond with a single line starting with FINAL_ANSWER: followed by only the final value (no JSON)."
        )

    def ask_llm(self, messages: List[Dict[str, str]]):
        assert self.cfg.model, 'model required for ask_llm'
        provider = (self.cfg.provider or 'ollama').lower()
        if provider == 'gemini':
            api_key = self.cfg.gemini_api_key or os.environ.get('GEMINI_API_KEY')
            if not api_key:
                raise RuntimeError('GEMINI_API_KEY not set in config or environment')
            return gemini_chat_stream(api_key, self.cfg.model, messages)
        # default to ollama
        return ollama_chat_stream(self.cfg.ollama, self.cfg.model, messages)

    def run(self, user_prompt: str) -> Dict[str, Any]:
        self.load_tools()
        tools_ctx = self.build_tools_context()
        plan_sys = self.make_prompt_plan(user_prompt)

        step = 0
        attempts = 0
        conversation: List[Dict[str, str]] = []
        # System priming: tools + protocol
        protocol = (
            "When calling a tool, output exactly one line with no extra text: "
            "TOOL_CALL {\"tool\":\"<name>\", \"input\": { ... }}\n"
            "When you are done, output a line starting with: FINAL_ANSWER: <your answer>."
        )
        conversation.append({'role': 'system', 'content': tools_ctx + '\n' + protocol})
        conversation.append({'role': 'system', 'content': plan_sys})
        conversation.append({'role': 'user', 'content': user_prompt})

        trace: List[Dict[str, Any]] = []
        final_answer: Optional[str] = None
        last_tool_output: Optional[Dict[str, Any]] = None

        while step < self.cfg.max_steps:
            step += 1
            attempts = 0
            # Ask LLM to either issue a tool call or a final answer
            events = self.ask_llm(conversation)
            kind, payload = detect_tool_call_and_final(events)
            if kind == 'final':
                final_answer = str(payload or '').strip()
                break
            if kind == 'tool' and isinstance(payload, dict):
                tool = payload.get('tool')
                input_obj = payload.get('input') or {}
                trace.append({'attempt': len(trace)+1, 'tool': tool, 'input': input_obj})
                output, error = invoke_tool_stream(self.cfg.server, tool, input_obj)
                if error:
                    # Inform the LLM and retry selection within this step
                    attempts += 1
                    conversation.append({'role': 'system', 'content': f"Tool '{tool}' failed: {json.dumps(error)}. Try another tool or adjust input."})
                    if attempts <= self.cfg.retry_per_step:
                        continue
                    # move to next step after exceeding retries
                    continue
                # success
                last_tool_output = output
                # hand result back and ask for final answer or next tool
                conversation.append({'role': 'tool', 'content': json.dumps({'tool': tool, 'output': output})})
                continue
            # No tool and no final -> ask explicitly for final
            conversation.append({'role': 'system', 'content': 'No tool call detected. Provide FINAL_ANSWER now.'})

        if final_answer is None:
            # One more pass to get final answer if we had a tool result
            if last_tool_output is not None:
                conversation.append({'role': 'system', 'content': 'Provide FINAL_ANSWER using the latest tool results.'})
                for ev in self.ask_llm(conversation):
                    msg = ev.get('message') or {}
                    delta = msg.get('content') or ''
                    if delta:
                        sys.stdout.write(delta)
                        sys.stdout.flush()
                        if 'FINAL_ANSWER:' in delta:
                            final_answer = delta.split('FINAL_ANSWER:', 1)[1].strip()
                            break

        if not final_answer and last_tool_output is not None:
            try:
                if isinstance(last_tool_output, dict):
                    # default to serializing structured output for visibility
                    final_answer = json.dumps(last_tool_output)
                else:
                    final_answer = str(last_tool_output)
            except Exception:
                final_answer = 'Done'

        return {
            'final_answer': final_answer,
            'trace': trace,
            'last_tool_output': last_tool_output,
        }
