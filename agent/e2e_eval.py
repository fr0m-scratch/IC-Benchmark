#!/usr/bin/env python3
"""End-to-end evaluation over fixed Top-K candidates for SLM vs LLM."""
import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from jsonschema import Draft7Validator, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


@dataclass
class ToolSpec:
    tool_id: str
    title: str
    description: str
    input_schema: Dict[str, Any]
    required: List[str]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_tools(path: str) -> Dict[str, ToolSpec]:
    specs = {}
    for row in load_jsonl(path):
        schema = row.get("input_schema") or {}
        required = schema.get("required") or []
        specs[row["tool_id"]] = ToolSpec(
            tool_id=row["tool_id"],
            title=row.get("title", ""),
            description=row.get("description", ""),
            input_schema=schema,
            required=required,
        )
    return specs


def load_candidates(path: str) -> Dict[str, Dict[str, Any]]:
    return {row["task_id"]: row for row in load_jsonl(path)}


SYSTEM_PROMPT = (
    "You are an API routing assistant. Your job is to pick the correct tool from the provided "
    "candidate list and produce strict JSON with the tool_id and arguments that satisfy the tool's "
    "schema. Do not invent tools or fields. Only respond with valid JSON."
)


def build_user_prompt(task: Dict[str, Any], candidates: List[Dict[str, Any]], tools: Dict[str, ToolSpec]) -> str:
    lines = [
        f"Task ID: {task['task_id']}",
        f"User request: {task['query']}",
        "Choose one tool from the list and provide arguments JSON that matches its required schema.",
        "Follow the schema exactly: use only required fields unless optional ones are clearly needed, respect types, and when an enum or allowed value list is given you must pick one of those values (e.g., priority must be one of {low, medium, high}).",
        "Tools:",
    ]
    for idx, cand in enumerate(candidates, start=1):
        tool = tools.get(cand["tool_id"])
        required = ", ".join(tool.required) if tool else "Unknown"
        lines.append(
            f"{idx}. tool_id={cand['tool_id']}\n"
            f"   title={tool.title if tool else 'UNKNOWN'}\n"
            f"   description={tool.description if tool else 'UNKNOWN'}\n"
            f"   required_fields=[{required}]"
        )
    lines.append(
        "Return JSON exactly in the form {\"tool_id\": <id>, \"arguments\": { ... }} with no extra text."
    )
    return "\n".join(lines)


def call_ollama(ollama_url: str,
                model: str,
                messages: List[Dict[str, str]],
                temperature: float,
                max_tokens: int) -> str:
    import urllib.request

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        ollama_url.rstrip("/") + "/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
        obj = json.loads(raw.decode("utf-8"))
    message = obj.get("message", {})
    return message.get("content", "")


class RateLimitError(RuntimeError):
    pass


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc)
    return "quota" in msg.lower() or "rate limit" in msg.lower() or "429" in msg


@retry(
    reraise=True,
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    retry=retry_if_exception_type(RateLimitError),
)
def call_gemini(api_key: str,
                model: str,
                messages: List[Dict[str, str]],
                temperature: float,
                max_tokens: int) -> str:
    import google.generativeai as genai
    from google.generativeai import types as genai_types

    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    user_parts = [m["content"] for m in messages if m["role"] != "system"]
    user_text = "\n\n".join(user_parts) or "Respond to the user."

    genai.configure(api_key=api_key)
    model_name = model if model.startswith("models/") else f"models/{model}"
    gen_model = genai.GenerativeModel(model_name)

    contents = [{
        "role": "user",
        "parts": [{"text": user_text}],
    }]

    try:
        response = gen_model.generate_content(
            contents,
            generation_config=genai_types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
    except Exception as exc:  # pragma: no cover - pass through structured errors
        if _is_quota_error(exc):
            raise RateLimitError(str(exc)) from exc
        raise

    # google-generativeai returns `.text` convenience property when the response contains text parts.
    return getattr(response, "text", "") or ""


def extract_json_candidate(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    cleaned = text.strip()
    if not cleaned:
        return None, "empty_response"
    # Try direct JSON
    try:
        return json.loads(cleaned), None
    except json.JSONDecodeError:
        pass
    # Fallback: search for JSON object
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        snippet = match.group(0)
        try:
            return json.loads(snippet), None
        except json.JSONDecodeError as exc:
            return None, f"json_decode_error: {exc}"
    return None, "no_json_found"


def build_validator(schema: Dict[str, Any]) -> Optional[Draft7Validator]:
    if not schema:
        return None
    try:
        return Draft7Validator(schema)
    except Exception:
        return None


def evaluate_task(task: Dict[str, Any],
                  candidates: List[Dict[str, Any]],
                  tools: Dict[str, ToolSpec],
                  provider: str,
                  model: str,
                  temperature: float,
                  max_tokens: int,
                  ollama_url: Optional[str],
                  gemini_key: Optional[str],
                  gemini_throttle: float = 0.0,
                  mock_response: Optional[str] = None) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(task, candidates, tools)},
    ]
    t_start = time.time()
    raw_response = ""
    error = None
    try:
        if mock_response is not None:
            raw_response = mock_response
        elif provider == "ollama":
            raw_response = call_ollama(ollama_url or "http://localhost:11434", model, messages, temperature, max_tokens)
        else:
            if gemini_throttle and gemini_throttle > 0:
                time.sleep(gemini_throttle)
            raw_response = call_gemini(gemini_key or "", model, messages, temperature, max_tokens)
    except Exception as exc:
        error = f"generation_error: {exc}"
    latency = time.time() - t_start

    parsed, parse_error = (None, None)
    if error is None:
        parsed, parse_error = extract_json_candidate(raw_response)
        if parse_error:
            error = parse_error

    result = {
        "task_id": task["task_id"],
        "query": task["query"],
        "gold": task["gold_tool_id"],
        "provider": provider,
        "model": model,
        "raw_response": raw_response,
        "latency_sec": latency,
        "selection_correct": False,
        "json_valid": False,
        "schema_valid": False,
        "pass_at_1": False,
        "error_type": error,
    }

    if parsed is None:
        return result

    tool_id = parsed.get("tool_id")
    arguments = parsed.get("arguments")
    result["predicted_tool"] = tool_id
    result["arguments"] = arguments

    if not isinstance(tool_id, str):
        result["error_type"] = "missing_tool_id"
        return result

    tool = tools.get(tool_id)
    if tool is None:
        result["error_type"] = "unknown_tool"
        return result

    result["selection_correct"] = tool_id == task["gold_tool_id"]

    if not isinstance(arguments, dict):
        result["error_type"] = "arguments_not_object"
        return result

    validator = build_validator(tool.input_schema)
    if validator is None:
        result["json_valid"] = True
        result["schema_valid"] = True
    else:
        try:
            validator.validate(arguments)
            result["json_valid"] = True
            result["schema_valid"] = True
        except ValidationError as exc:
            result["error_type"] = f"schema_error: {exc.message}"
            return result

    result["error_type"] = None
    result["pass_at_1"] = result["selection_correct"] and result["schema_valid"]
    return result


def summarise_results(rows: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
    n = len(rows) or 1
    acc = sum(1.0 for r in rows if r.get("selection_correct")) / n
    json_rate = sum(1.0 for r in rows if r.get("json_valid")) / n
    schema_rate = sum(1.0 for r in rows if r.get("schema_valid")) / n
    pass_rate = sum(1.0 for r in rows if r.get("pass_at_1")) / n
    return [
        ("tasks", len(rows)),
        ("selection_accuracy", acc),
        ("json_valid_rate", json_rate),
        ("schema_valid_rate", schema_rate),
        ("pass@1", pass_rate),
    ]


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on fixed Top-K tool candidates.")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--tools", required=True)
    parser.add_argument("--candidates", required=True, help="JSONL with candidate list per task")
    parser.add_argument("--provider", choices=["ollama", "gemini"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--gemini-api-key")
    parser.add_argument("--gemini-throttle", type=float, default=0.0, help="Seconds to sleep before each Gemini call to avoid rate limits")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--tag", default="run")
    parser.add_argument("--limit", type=int, help="Limit number of tasks for smoke testing")
    parser.add_argument("--mock-responses", help="Optional JSONL mapping task_id to synthetic model outputs")
    args = parser.parse_args()

    if args.provider == "ollama" and not args.model:
        raise SystemExit("--model required for Ollama")
    if args.provider == "gemini" and not args.gemini_api_key and not args.mock_responses:
        raise SystemExit("--gemini-api-key required for Gemini provider")

    os.makedirs(args.outdir, exist_ok=True)

    tools = load_tools(args.tools)
    candidates_map = load_candidates(args.candidates)
    tasks = load_jsonl(args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]

    rows: List[Dict[str, Any]] = []
    mock_map = {}
    if args.mock_responses:
        mock_map = {row["task_id"]: row.get("response") for row in load_jsonl(args.mock_responses)}

    missing = [t["task_id"] for t in tasks if t["task_id"] not in candidates_map]
    if missing:
        raise SystemExit(f"Candidates missing for tasks: {missing}")

    for task in tasks:
        candidate_entry = candidates_map[task["task_id"]]
        cand_list = candidate_entry.get("candidates") or []
        row = evaluate_task(
            task,
            candidates=cand_list,
            tools=tools,
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_output_tokens,
            ollama_url=args.ollama_url,
            gemini_key=args.gemini_api_key,
            gemini_throttle=args.gemini_throttle,
            mock_response=mock_map.get(task["task_id"]),
        )
        row["tag"] = args.tag
        row["candidate_set"] = os.path.basename(args.candidates)
        rows.append(row)

    per_task_path = os.path.join(args.outdir, f"per_task_{args.tag}.jsonl")
    with open(per_task_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = os.path.join(args.outdir, f"summary_{args.tag}.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        f.write("metric,value\n")
        for metric, value in summarise_results(rows):
            if isinstance(value, float):
                f.write(f"{metric},{value:.3f}\n")
            else:
                f.write(f"{metric},{value}\n")

    print(f"Saved per-task results to {per_task_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
