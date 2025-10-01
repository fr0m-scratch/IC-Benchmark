import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.ic_wrap import describe_wrapper_modes, wrap_schema


SCHEMA = {
    "type": "object",
    "required": ["name", "count"],
    "properties": {
        "name": {"type": "string"},
        "count": {"type": "integer"},
        "mode": {"type": "string", "enum": ["fast", "safe"]},
    },
}


def test_flat_wrap_keeps_required_only():
    wrapped = wrap_schema(SCHEMA, "flat")
    assert set(wrapped["properties"].keys()) == {"name", "count"}
    assert all("examples" in prop for prop in wrapped["properties"].values())


def test_hard_wrap_adds_constraints():
    wrapped = wrap_schema(SCHEMA, "hard")
    assert set(wrapped["required"]) == {"name", "count", "mode"}
    assert "allOf" in wrapped


def test_describe_wrapper_modes_contains_expected_keys():
    modes = describe_wrapper_modes()
    assert {"none", "flat", "hard"}.issubset(modes.keys())
