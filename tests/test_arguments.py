import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.arguments import (
    describe_schema,
    find_missing_required,
    full_argument_prompt,
    should_use_slot_mode,
    slot_fill_prompt,
    synthesise_minimal_arguments,
    validate_arguments,
)


SCHEMA = {
    "type": "object",
    "required": ["name", "count"],
    "properties": {
        "name": {"type": "string", "description": "Display name"},
        "count": {"type": "integer", "minimum": 1},
        "mode": {"type": "string", "enum": ["fast", "safe"]},
    },
}


def test_describe_schema_lists_required():
    description = describe_schema(SCHEMA)
    assert "name" in description
    assert "required" in description


def test_validate_arguments_flags_missing():
    ok, _ = validate_arguments(SCHEMA, {"name": "demo", "count": 2})
    assert ok
    ok, message = validate_arguments(SCHEMA, {"name": "demo"})
    assert not ok
    assert "count" in message


def test_slot_fill_prompt_reports_missing():
    prompt = slot_fill_prompt(SCHEMA, {"name": "demo"})
    assert "Missing required fields" in prompt
    assert "count" in prompt


def test_should_use_slot_mode_threshold():
    assert should_use_slot_mode(0.7, 0.5)
    assert not should_use_slot_mode(0.3, 0.5)


def test_full_argument_prompt_contains_task():
    prompt = full_argument_prompt(SCHEMA, "Update inventory")
    assert "Update inventory" in prompt
    assert "Schema" in prompt


def test_find_missing_required():
    missing = find_missing_required(SCHEMA, {"name": "demo"})
    assert missing == ["count"]


def test_synthesise_minimal_arguments_populates_required():
    args = synthesise_minimal_arguments(SCHEMA)
    assert set(args.keys()) == {"name", "count"}
