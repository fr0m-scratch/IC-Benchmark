#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
from typing import Dict, List

from jsonschema import ValidationError, validate

from providers import FireworksChat, GeminiGen, OllamaChat
from prompts import SYSTEM, format_user


def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _top_required(schema: Dict) -> List[str]:
    if isinstance(schema, dict) and isinstance(schema.get("required"), list):
        return schema["required"]
    return []


def eval_task(task: Dict, candidates: List, tools_by_id: Dict[str, Dict], client) -> Dict:
    cand_meta = []
    for item in candidates:
        tid = item if isinstance(item, str) else item.get("tool_id")
        if not tid:
            continue
        tool = tools_by_id[tid]
        cand_meta.append({
            "tool_id": tid,
            "title": tool.get("title", ""),
            "short_desc": tool.get("short_desc", tool.get("description", "")),
            "required": _top_required(tool.get("input_schema", {})),
            "schema": tool.get("input_schema", {}),
        })
    user_prompt = format_user(task["query"], cand_meta, len(cand_meta))
    response = client.chat(SYSTEM, user_prompt)

    chosen = response.get("tool_id")
    arguments = response.get("arguments", {})

    select_right = int(chosen == task["gold"])
    schema_valid = 0
    if chosen and chosen in tools_by_id:
        try:
            validate(arguments, tools_by_id[chosen].get("input_schema", {}))
            schema_valid = 1
        except ValidationError:
            schema_valid = 0
    return {
        "pred": response,
        "select_right": select_right,
        "schema_valid": schema_valid,
        "pass@1": int(select_right and schema_valid),
    }


def main():
    parser = argparse.ArgumentParser(description="Offline apples-to-apples evaluator")
    parser.add_argument("--candidates_jsonl", required=True)
    parser.add_argument("--tools_jsonl", required=True)
    parser.add_argument("--model", choices=["slm", "llm", "fireworks"], required=True)
    parser.add_argument("--slm_name", default="qwen2.5:7b-instruct")
    parser.add_argument("--llm_name", default="models/gemini-2.5-flash")
    parser.add_argument("--fireworks_name", default="accounts/fireworks/models/glm-4p5-air")
    parser.add_argument("--out_jsonl", required=True)
    args = parser.parse_args()

    tools = load_jsonl(args.tools_jsonl)
    tools_by_id = {t["tool_id"]: t for t in tools}
    candidates = load_jsonl(args.candidates_jsonl)

    if args.model == "slm":
        client = OllamaChat(model=args.slm_name)
    elif args.model == "llm":
        client = GeminiGen(model=args.llm_name)
    else:
        client = FireworksChat(model=args.fireworks_name)

    with open(args.out_jsonl, "w", encoding="utf-8") as out:
        for entry in candidates:
            result = eval_task(
                {"task_id": entry["task_id"], "query": entry["query"], "gold": entry["gold"]},
                entry["candidates"],
                tools_by_id,
                client,
            )
            out.write(json.dumps({**entry, **result}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
