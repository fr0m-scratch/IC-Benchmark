import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.ic_score import ToolRecord, score_tool
from bench.retrieval import RetrievalConfig, ToolRetriever


def _make_tool(tool_id: str, name: str, description: str, schema: dict) -> ToolRecord:
    return ToolRecord(
        tool_id=tool_id,
        name=name,
        description=description,
        input_schema=schema,
        output_schema=None,
        server_id="srv",
        server_name="srv",
        raw={"tool_id": tool_id, "input_schema": schema},
    )


def test_retriever_scores_and_penalises_complexity():
    simple_schema = {
        "type": "object",
        "required": ["email"],
        "properties": {"email": {"type": "string", "format": "email"}},
    }
    complex_schema = {
        "type": "object",
        "required": ["email", "password", "role"],
        "properties": {
            "email": {"type": "string", "format": "email"},
            "password": {"type": "string", "minLength": 8},
            "role": {"type": "string", "enum": ["admin", "viewer", "guest"]},
        },
    }
    simple_tool = _make_tool("srv::simple", "create_user", "Create a user", simple_schema)
    complex_tool = _make_tool("srv::complex", "create_user_secure", "Create a user with password and role", complex_schema)

    ic_results = {
        simple_tool.tool_id: score_tool(simple_tool),
        complex_tool.tool_id: score_tool(complex_tool),
    }

    retriever = ToolRetriever(RetrievalConfig(k=2, ic_penalty_mu=0.5, redundancy_penalty_nu=0.0))
    retriever.build([simple_tool, complex_tool], ic_results)

    results = retriever.query("create a user account with email")
    assert len(results) == 2
    # Penalised score favours the simpler tool
    assert results[0].tool.tool_id == "srv::simple"
    assert results[0].score > results[1].score


def test_retriever_redundancy_penalty():
    schema = {
        "type": "object",
        "required": ["query"],
        "properties": {"query": {"type": "string"}},
    }
    tool_a = _make_tool("srv::a", "search_docs", "Search documents", schema)
    tool_b = _make_tool("srv::b", "search_files", "Search documents", schema)
    ic_results = {tool_a.tool_id: score_tool(tool_a), tool_b.tool_id: score_tool(tool_b)}

    retriever = ToolRetriever(RetrievalConfig(k=2, ic_penalty_mu=0.0, redundancy_penalty_nu=0.5))
    retriever.build([tool_a, tool_b], ic_results)
    results = retriever.query("search documents")

    assert len(results) == 2
    # Redundancy penalty applied (scores drop below similarity)
    for res in results:
        assert res.redundancy >= 0
        assert res.score <= res.similarity
