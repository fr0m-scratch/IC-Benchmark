import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.ic_score import (
    SetICResult,
    ToolICResult,
    ToolRecord,
    compute_ic_set,
    score_server,
    score_tool,
)


def _make_tool_record(tool_id: str, name: str, description: str, schema: dict, server_id: str = "srv") -> ToolRecord:
    return ToolRecord(
        tool_id=tool_id,
        name=name,
        description=description,
        input_schema=schema,
        output_schema=None,
        server_id=server_id,
        server_name=server_id,
        raw={"tool_id": tool_id, "input_schema": schema},
    )


def test_score_tool_counts_expected_features():
    schema = {
        "type": "object",
        "required": ["name", "threshold"],
        "properties": {
            "name": {
                "type": "string",
                "description": "User-facing name",
                "examples": ["primary"],
            },
            "threshold": {
                "type": "number",
                "minimum": 0,
                "maximum": 10,
                "multipleOf": 0.5,
            },
            "mode": {
                "type": "string",
                "enum": ["fast", "safe"],
            },
            "options": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "items": {"type": "integer"},
            },
        },
    }
    tool = _make_tool_record("srv::tool", "tool", "Synthetic tool", schema)
    result = score_tool(tool)

    assert math.isfinite(result.score)
    assert result.features["total_fields"] == 4
    assert result.features["required_fields"] == 2
    assert result.features["enum_bits"] > 0
    assert result.features["numeric_bits"] > 0
    assert result.features["array_cost"] > 0
    assert "log_leaf_fields" in result.features
    # Components mirror the weighted features
    assert set(result.components.keys()).issuperset({
        "log_leaf_fields",
        "required_ratio",
        "max_depth",
    })


def test_compute_ic_set_reflects_redundancy():
    schema = {"type": "object", "properties": {}}
    base_tool = _make_tool_record("srv::one", "one", "First tool", schema)
    redundant_tool = _make_tool_record("srv::two", "two", "First tool", schema)
    diverse_tool = _make_tool_record("srv::three", "three", "Second distinct capability", schema, server_id="srv-b")

    res_base = score_tool(base_tool)
    res_redundant = score_tool(redundant_tool)
    res_diverse = score_tool(diverse_tool)

    set_redundant: SetICResult = compute_ic_set([res_base, res_redundant])
    set_diverse: SetICResult = compute_ic_set([res_base, res_diverse])

    assert set_redundant.features["redundancy"] >= set_diverse.features["redundancy"]
    assert set_diverse.features["server_diversity"] > set_redundant.features["server_diversity"]


def test_score_server_aggregates_tools():
    schema = {"type": "object", "properties": {}}
    tool_a = score_tool(_make_tool_record("srv::a", "a", "Authenticate users", schema))
    tool_b = score_tool(_make_tool_record("srv::b", "b", "List accounts", schema))

    result = score_server(
        server_id="srv",
        server_name="srv",
        tool_results=[tool_a, tool_b],
        server_meta={"description": "Handles auth and pagination."},
    )

    assert result.features["tool_count"] == 2
    assert result.features["statefulness"] > 0  # picks up auth keyword
    assert math.isfinite(result.score)
    assert set(result.tool_ids) == {"srv::a", "srv::b"}
