"""
Language model-based tool retrieval (minimal, research-friendly refactor).

Exposes a single function:
    retrieve_lm(request, tool_list, model_config, topk) -> list[dict]

Inputs
- request: natural language request string.
- tool_list: list of tool dictionaries (e.g., from server-crawling/tools_parsed.json).
  The model will be shown only per-tool fields without any server information:
    - name: Unique identifier for the tool (falls back to tool_name if present)
    - title: Optional human-readable name (if present)
    - description: Human-readable description (falls back to tool_description)
    - inputSchema: JSON Schema defining expected parameters (falls back to input_schema)
    - outputSchema: Optional JSON Schema defining expected output structure (if present)
    - annotations: Optional properties describing tool behavior (if present)
- model_config: dict with at least a model identifier, e.g. {"model_id": "Qwen/Qwen2.5-7B-Instruct"}.
  Other vLLM engine keys may be included (e.g., dtype, tensor_parallel_size, etc.).
- topk: number of tools to select (>=1).

Returns
- A list of up to `topk` tool dictionaries from the original `tool_list`, in the
  order returned by the model. If the model returns fewer, the list may be shorter.

This module intentionally keeps logic simple and removes safety scaffolding.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional


def _normalize_tool_for_prompt(tool: Dict[str, Any]) -> Dict[str, Any]:
    """Produce a server-free representation for prompting.

    Maps common keys from tools_parsed.json to the desired schema.
    """
    name = tool.get("name") or tool.get("tool_name") or ""
    out: Dict[str, Any] = {"name": name}

    # Optional fields if available
    title = tool.get("title")
    if title:
        out["title"] = title

    desc = tool.get("description") or tool.get("tool_description")
    if desc:
        out["description"] = desc

    in_schema = tool.get("inputSchema") or tool.get("input_schema")
    if isinstance(in_schema, dict):
        out["inputSchema"] = in_schema

    out_schema = tool.get("outputSchema") or tool.get("output_schema")
    if isinstance(out_schema, dict):
        out["outputSchema"] = out_schema

    annotations = tool.get("annotations")
    if isinstance(annotations, (dict, list)):
        out["annotations"] = annotations

    # Map non-spec extras from the dataset when present
    # tools_parsed.json commonly includes parameter_descriptions; surface it
    # under annotations if not already provided.
    if "annotations" not in out and isinstance(tool.get("parameter_descriptions"), (dict, list)):
        out["annotations"] = {"parameter_descriptions": tool.get("parameter_descriptions")}

    return out


def _format_tools_for_prompt(tools: Iterable[Dict[str, Any]]) -> str:
    """Format tools for the prompt as newline-separated JSON objects."""
    return "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)


def _build_selection_prompt(request: str, tool_block: str, top_k: int) -> str:
    """Prompt asking the model to select top-K tools by name only.

    Minimal deviation from the prior prompt: removes server references and
    requests output as a JSON list of tool names.
    """
    return f"""
You are given a list of available tools and a user request.
Select the MOST appropriate tool(s) that best match the user's request.

Available tools (one per line, JSON objects):
{tool_block}

User request:
{request}

Rules:
- Return exactly {top_k} tools in order of preference as JSON:
  {{"selected_tools": ["tool_name_a", "tool_name_b", ...]}}
- If fewer than {top_k} are suitable, return as many as you can (non-empty).
- Do NOT include any other fields or commentary.
""".strip()


def _parse_selected_tools(text: str) -> List[str]:
    """Parse model output for {"selected_tools": [..]} with a small fallback."""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("selected_tools"), list):
            return [str(x) for x in obj.get("selected_tools")]
    except Exception:
        pass
    # Fallback: extract a JSON-like list of quoted strings under selected_tools
    import re
    m = re.search(r"\{\s*\"selected_tools\"\s*:\s*\[(.*?)\]", text, re.DOTALL)
    if m:
        arr = m.group(1)
        tools = re.findall(r'\"([^\"]+)\"', arr)
        return [str(x) for x in tools]
    return []


class _VLLMModel:
    """Light wrapper around vLLM for text completion."""

    def __init__(self, model_configs: Optional[Dict[str, Any]] = None) -> None:
        try:
            from vllm import LLM, SamplingParams  # type: ignore
        except Exception as e:  # pragma: no cover - import-time dependency
            raise RuntimeError("vLLM is required. Please install vllm.") from e
        self._SamplingParams = SamplingParams

        cfg = dict(model_configs or {})

        # Minimal allowlist of vLLM engine args
        allowed_keys = {
            "max_model_len",
            "gpu_memory_utilization",
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "dtype",
            "quantization",
            "kv_cache_dtype",
            "rope_scaling",
            "seed",
            "enforce_eager",
            "tokenizer_mode",
            "max_seq_len_to_capture",
            "download_dir",
            "load_format",
            "compressed_tensors",
            "max_num_seqs",
            "model",
            "trust_remote_code"
        }
        llm_kwargs = {k: cfg[k] for k in allowed_keys if k in cfg}
        assert "model" in llm_kwargs, "model_config must include 'model' for vLLM"
        self._llm = LLM(**llm_kwargs)

    def complete(self, prompt: str, *, temperature: float = 0.2, top_p: float = 0.95) -> str:
        # Do not set max_tokens explicitly; rely on vLLM defaults/config.
        params = self._SamplingParams(temperature=temperature, top_p=top_p)
        out = self._llm.generate([prompt], params)
        if not out or not out[0].outputs:
            return ""
        return (out[0].outputs[0].text or "").strip()


def retrieve_lm(
    request: str,
    tool_list: List[Dict[str, Any]],
    model_config: Dict[str, Any],
    topk: int,
) -> List[Dict[str, Any]]:
    """Select top-K tools for a request using an LLM.

    - Builds a server-free tool presentation using only: name, title, description,
      inputSchema, outputSchema, annotations (when present).
    - Prompts a vLLM-backed model to return JSON {"selected_tools": [names...]}
    - Returns the corresponding tool dictionaries from the provided `tool_list`.
    """
    if not isinstance(topk, int) or topk <= 0:
        topk = 1

    # Identify model id
    model_id = (
        model_config.get("model_id")
        or model_config.get("id")
        or model_config.get("repo_id")
    )
    if not model_id:
        raise ValueError("model_config must contain 'model_id' (or 'id'/'repo_id').")

    # Build tool prompt block
    normalized = [_normalize_tool_for_prompt(t) for t in tool_list]
    tool_block = _format_tools_for_prompt(normalized)
    prompt = _build_selection_prompt(request, tool_block, topk)

    # Inference
    model = _VLLMModel(model_id, model_configs=model_config)
    raw = model.complete(prompt)
    chosen_names = _parse_selected_tools(raw)

    # Map back to original tool dicts; prefer exact name, fall back to tool_name
    index: Dict[str, Dict[str, Any]] = {}
    for t in tool_list:
        key = t.get("name") or t.get("tool_name")
        if isinstance(key, str) and key and key not in index:
            index[key] = t

    selected: List[Dict[str, Any]] = []
    for n in chosen_names:
        t = index.get(n)
        if t is not None and t not in selected:
            selected.append(t)
        if len(selected) >= topk:
            break

    return selected
