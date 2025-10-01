import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.providers import ProviderConfig, _extract_json, build_provider


def test_extract_json_handles_trailing_commas():
    text = "Here is JSON: {\"a\": 1,}\nthanks"
    data = _extract_json(text)
    assert data == {"a": 1}


def test_build_provider_constructs_ollama():
    config = ProviderConfig(provider="ollama", model="test")
    provider = build_provider(config)
    assert provider.__class__.__name__ == "OllamaProvider"
