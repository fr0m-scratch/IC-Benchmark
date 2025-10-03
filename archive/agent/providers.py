import json
import os
import re
from typing import Any, Dict

import requests


def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    snippet = match.group(0)
    for fixer in (lambda s: s, _fix_trailing_commas):
        try:
            return json.loads(fixer(snippet))
        except Exception:
            continue
    return {}


def _fix_trailing_commas(payload: str) -> str:
    payload = re.sub(r",\s*\}", "}", payload)
    payload = re.sub(r",\s*\]", "]", payload)
    return payload


class OllamaChat:
    def __init__(self, model: str, host: str = None, timeout: int = 120):
        self.model = model
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.timeout = timeout

    def chat(self, system: str, user: str) -> Dict[str, Any]:
        url = f"{self.host.rstrip('/')}/api/chat"
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        text = resp.json().get("message", {}).get("content", "")
        return _extract_json(text)


class GeminiGen:
    def __init__(self, model: str = "models/gemini-2.5-flash", api_key: str = None, timeout: int = 120):
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError("Missing GEMINI_API_KEY")

    def chat(self, system: str, user: str) -> Dict[str, Any]:
        import google.generativeai as genai
        from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

        genai.configure(api_key=self.api_key)
        prompt = f"{system}\n\n{user}"
        model = genai.GenerativeModel(self.model)

        def _is_quota_error(exc: Exception) -> bool:
            msg = str(exc).lower()
            return any(token in msg for token in ("quota", "rate", "429"))

        @retry(
            reraise=True,
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=2, min=2, max=60),
            retry=retry_if_exception(_is_quota_error),
        )
        def _generate():
            return model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,
                },
            )

        response = _generate()
        text = getattr(response, "text", "") or ""
        return _extract_json(text)


class FireworksChat:
    def __init__(self, model: str = "accounts/fireworks/models/glm-4p5-air", api_key: str = None, timeout: int = 120):
        self.model = model
        self.api_key = api_key or os.getenv("FIREWORKS_API_KEY", "")
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError("Missing FIREWORKS_API_KEY")

    def chat(self, system: str, user: str) -> Dict[str, Any]:
        from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "top_p": 1,
            "top_k": 40,
            "temperature": 0.6,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        def _is_quota_error(exc: Exception) -> bool:
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                return exc.response.status_code == 429
            return False

        @retry(
            reraise=True,
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=2, min=2, max=60),
            retry=retry_if_exception(_is_quota_error),
        )
        def _call():
            resp = requests.post(
                "https://api.fireworks.ai/inference/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp

        resp = _call()
        data = resp.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return _extract_json(text)
