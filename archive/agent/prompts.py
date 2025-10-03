SYSTEM = "You are a tool-using assistant. Read the task and the candidate tools. Choose ONE tool and produce STRICT JSON."

USER_TMPL = """Task: {task}

Candidate tools (Top-{k}):
{tool_lines}

Rules:
- You MUST pick exactly one of the listed tool_id.
- Output STRICT JSON with keys: \"tool_id\", \"arguments\".
- \"arguments\" must satisfy the tool's JSON Schema (required fields present; types match; enum values must be chosen from the allowed list).
JSON ONLY:
"""

def format_user(task_text, cand_meta, k):
    lines = []
    for idx, cm in enumerate(cand_meta, 1):
        required = ", ".join(cm.get("required", [])) or "(none)"
        title = cm.get("title", "")
        desc = cm.get("short_desc", cm.get("description", ""))
        lines.append(f"{idx}) {cm['tool_id']}: {title}. {desc} REQUIRED: {required}")
    return USER_TMPL.format(task=task_text, k=k, tool_lines="\n".join(lines))
