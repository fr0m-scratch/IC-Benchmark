"""Unified chat provider abstractions for the benchmark agent."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence

import requests

JSON_SNIPPET_RE = re.compile(r"\{[\s\S]*\}")


class ChatError(RuntimeError):
    """Raised when a provider request fails."""


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatResult:
    text: str
    raw: Dict[str, Any]
    provider: str

    def json(self) -> Dict[str, Any]:
        data = _extract_json(self.text)
        if data:
            return data
        return {}


class ChatProvider(Protocol):
    def generate(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResult:
        ...


def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    match = JSON_SNIPPET_RE.search(text)
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


@dataclass
class ProviderConfig:
    provider: str
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: int = 120


class OllamaProvider:
    def __init__(self, config: ProviderConfig):
        self.model = config.model
        self.base_url = config.base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.timeout = config.timeout

    def generate(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResult:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        url = f"{self.base_url.rstrip('/')}/api/chat"
        resp = requests.post(url, json=payload, timeout=self.timeout)
        if resp.status_code >= 400:
            raise ChatError(f"Ollama error {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        text = data.get("message", {}).get("content", "")
        return ChatResult(text=text, raw=data, provider="ollama")


class GeminiProvider:
    def __init__(self, config: ProviderConfig):
        self.model = config.model or "models/gemini-2.5-flash"
        self.api_key = config.api_key or os.getenv("GEMINI_API_KEY")
        self.timeout = config.timeout
        if not self.api_key:
            raise ChatError("Missing GEMINI_API_KEY environment variable")

    def generate(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResult:
        try:
            import google.generativeai as genai
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ChatError("google-generativeai not installed") from exc

        genai.configure(api_key=self.api_key)
        prompt = "\n".join(m.content for m in messages)
        model = genai.GenerativeModel(self.model)
        response = model.generate_content(prompt, generation_config={"temperature": kwargs.get("temperature", 0.1)})
        text = getattr(response, "text", "") or ""
        raw = getattr(response, "to_dict", lambda: {} )()
        return ChatResult(text=text, raw=raw, provider="gemini")


class FireworksProvider:
    def __init__(self, config: ProviderConfig):
        self.model = config.model
        self.api_key = config.api_key or os.getenv("FIREWORKS_API_KEY")
        self.timeout = config.timeout
        if not self.api_key:
            raise ChatError("Missing FIREWORKS_API_KEY environment variable")

    def generate(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResult:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.4),
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        url = kwargs.get("base_url") or "https://api.fireworks.ai/inference/v1/chat/completions"
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
        if resp.status_code >= 400:
            raise ChatError(f"Fireworks error {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return ChatResult(text=text, raw=data, provider="fireworks")


def build_provider(config: ProviderConfig) -> ChatProvider:
    provider_name = config.provider.lower()
    if provider_name == "ollama":
        return OllamaProvider(config)
    if provider_name == "gemini":
        return GeminiProvider(config)
    if provider_name == "fireworks":
        return FireworksProvider(config)
    raise ChatError(f"Unsupported provider '{config.provider}'")


__all__ = [
    "ChatError",
    "ChatMessage",
    "ChatProvider",
    "ChatResult",
    "ProviderConfig",
    "build_provider",
    "_extract_json",
]
