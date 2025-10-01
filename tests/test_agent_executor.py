import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.ic_score import ToolRecord, score_tool
from bench.agent_executor import AgentConfig, AgentExecutor, InvocationResult
from bench.metrics import MetricsTracker
from bench.providers import ChatResult
from bench.retrieval import RetrievalConfig, ToolRetriever


def _make_tool(schema):
    return ToolRecord(
        tool_id="srv::demo",
        name="demo_tool",
        description="Demo tool",
        input_schema=schema,
        output_schema=None,
        server_id="srv",
        server_name="srv",
        raw={"tool_id": "srv::demo", "input_schema": schema},
    )


class StubProvider:
    def __init__(self, responses):
        self.responses = responses

    def generate(self, messages, **kwargs):
        text = self.responses.pop(0)
        return ChatResult(text=text, raw={}, provider="stub")


class StubInvoker:
    def __init__(self, success=True):
        self.success = success

    def invoke(self, tool_name, arguments):
        if not self.success:
            return InvocationResult(False, {}, {"code": "invalid_input", "message": "bad"}, "error", 0.0)
        return InvocationResult(True, {"echo": arguments}, None, "ok", 42.0)


def _build_retriever(tool, ic_results):
    retriever = ToolRetriever(RetrievalConfig(k=1, ic_penalty_mu=0.0, redundancy_penalty_nu=0.0))
    retriever.build([tool], ic_results)
    return retriever


def test_agent_executor_basic_flow(tmp_path):
    schema = {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}
    tool = _make_tool(schema)
    ic_results = {tool.tool_id: score_tool(tool)}
    retriever = _build_retriever(tool, ic_results)
    provider = StubProvider(["{\"name\": \"Alice\"}"])
    invoker = StubInvoker()

    executor = AgentExecutor(AgentConfig(max_steps=1, gate_threshold=-1.0), provider, retriever, ic_results, invoker)
    result = executor.run("Say hello")

    assert "Alice" in result["final_answer"]
    assert result["metrics"]["calls"] == 1


def test_agent_executor_slot_fill(tmp_path):
    schema = {
        "type": "object",
        "required": ["items"],
        "properties": {
            "items": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string"},
            }
        },
    }
    tool = _make_tool(schema)
    ic_score = score_tool(tool)
    # force high IC to trigger slot mode
    ic_results = {tool.tool_id: ic_score}
    retriever = _build_retriever(tool, ic_results)
    provider = StubProvider([
        "{}",  # initial attempt fails validation
        "{\"slot\": \"items\", \"value\": [\"foo\"]}",
    ])
    invoker = StubInvoker()

    config = AgentConfig(max_steps=1, slot_threshold=0.0)
    executor = AgentExecutor(config, provider, retriever, ic_results, invoker)
    result = executor.run("Fetch items")

    assert "foo" in result["final_answer"]
