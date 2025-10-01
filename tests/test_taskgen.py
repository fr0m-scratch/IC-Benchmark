import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.ic_score import ToolRecord
from bench.taskgen import DependencyGraph, TaskGenerator, _parse_tiers


def _make_tool(tool_id: str, name: str, description: str, schema: dict, server_id: str = "srv") -> ToolRecord:
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


def test_dependency_graph_pairs_list_and_assign():
    create_schema = {
        "type": "object",
        "required": ["email"],
        "properties": {
            "email": {"type": "string", "format": "email"},
            "name": {"type": "string"},
        },
    }
    assign_schema = {
        "type": "object",
        "required": ["user_id"],
        "properties": {
            "user_id": {"type": "string", "description": "Identifier returned from user creation"},
            "role": {"type": "string", "enum": ["admin", "viewer"]},
        },
    }
    create_tool = _make_tool("srv::create_user", "create_user", "Create a user account and return user_id", create_schema)
    assign_tool = _make_tool("srv::assign_role", "assign_role", "Assign a role to an existing user, requires user_id", assign_schema)

    deps = DependencyGraph([create_tool, assign_tool]).dependencies_for("srv::assign_role")
    assert deps, "dependency inference should detect upstream tool"
    assert deps[0].source_tool_id == "srv::create_user"
    assert deps[0].field == "user_id"


def test_task_generator_produces_generic_and_fuzzy_records():
    schema = {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {"type": "string", "description": "Query string", "examples": ["hello world"]},
            "limit": {"type": "integer", "minimum": 1, "maximum": 5},
        },
    }
    tool = _make_tool("srv::search", "search", "Perform catalog search", schema)
    generator = TaskGenerator([tool], random.Random(0))
    generic, fuzzy = generator.generate(tiers=["realistic"], limit=1, shuffle=False)

    assert len(generic) == 1
    assert len(fuzzy) == 1
    generic_record = generic[0]
    fuzzy_record = fuzzy[0]

    assert generic_record["tool"] == "srv::search"
    assert generic_record["input"]["query"].startswith("hello")
    assert "expect" in fuzzy_record and fuzzy_record["expect"]["tool"] == "srv::search"


def test_parse_tiers_handles_defaults_and_validation():
    assert _parse_tiers(None) == ["realistic"]
    assert _parse_tiers("flat, hard") == ["flat", "hard"]
    try:
        _parse_tiers("unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for unknown tier")
