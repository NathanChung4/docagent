"""Agent loop tests with a fake Anthropic client.

The shared fakes in `tests/_anthropic_fakes.py` script multi-iteration responses
so tests can assert on streaming order, tool dispatch, tool_result wiring, and
QueryLog contents.
"""

from __future__ import annotations

from typing import Any

from knowledge_rag.agent import (
    Agent,
    AgentResult,
    ToolCallEvent,
    ToolResultEvent,
    _build_tools_payload,
)
from knowledge_rag.tools.base import Tool, ToolValidationError
from tests._anthropic_fakes import (
    FakeClient,
    FakeStream,
    text_block,
    tool_use_block,
    usage,
)

# --- fake tools --------------------------------------------------------------


class _EchoTool(Tool):
    name = "echo"
    description = "Echo back the supplied message."
    input_schema = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    }

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"echoed": kwargs.get("message", "")}


class _AlwaysFailsTool(Tool):
    name = "always_fails"
    description = "Always raises ToolValidationError."
    input_schema = {"type": "object", "properties": {}, "required": []}

    def run(self, **kwargs: Any) -> dict[str, Any]:
        raise ToolValidationError("nope")


# --- payload builder ---------------------------------------------------------


def test_build_tools_payload_caches_last_tool() -> None:
    payload = _build_tools_payload([_EchoTool(), _AlwaysFailsTool()])
    assert len(payload) == 2
    assert "cache_control" not in payload[0]
    assert payload[1]["cache_control"] == {"type": "ephemeral"}
    # Both schemas must include name + description + input_schema.
    for entry in payload:
        assert {"name", "description", "input_schema"} <= entry.keys()


def test_build_tools_payload_handles_empty() -> None:
    assert _build_tools_payload([]) == []


# --- direct-answer path (no tool use) ----------------------------------------


def test_agent_streams_text_when_no_tool_use() -> None:
    client = FakeClient(
        [
            FakeStream(
                tokens=["Hello", " ", "world"],
                usage=usage(),
                content=[text_block("Hello world")],
                stop_reason="end_turn",
            )
        ]
    )
    agent = Agent(tools=[_EchoTool()], client=client)

    items = list(agent.stream("hi", []))
    text_items = [i for i in items if isinstance(i, str)]
    results = [i for i in items if isinstance(i, AgentResult)]

    assert text_items == ["Hello", " ", "world"]
    assert len(results) == 1
    res = results[0]
    assert res.answer == "Hello world"
    assert res.iterations == 1
    assert res.query_log.tool_calls == []
    assert res.query_log.first_token_ms > 0


# --- tool dispatch path ------------------------------------------------------


def test_agent_dispatches_tool_then_streams_final_answer() -> None:
    """First iteration: text intro + tool_use. Second iteration: final answer."""
    iter1 = FakeStream(
        tokens=["Let me check."],
        usage=usage(input_tokens=20, output_tokens=8),
        content=[
            text_block("Let me check."),
            tool_use_block("echo", {"message": "hi"}, id_="tu_xyz"),
        ],
        stop_reason="tool_use",
    )
    iter2 = FakeStream(
        tokens=["Done", "."],
        usage=usage(input_tokens=30, output_tokens=4),
        content=[text_block("Done.")],
        stop_reason="end_turn",
    )
    tool = _EchoTool()
    client = FakeClient([iter1, iter2])
    agent = Agent(tools=[tool], client=client)

    items = list(agent.stream("call echo", []))

    text_items = [i for i in items if isinstance(i, str)]
    tool_calls = [i for i in items if isinstance(i, ToolCallEvent)]
    tool_results = [i for i in items if isinstance(i, ToolResultEvent)]
    finals = [i for i in items if isinstance(i, AgentResult)]

    # Streaming order: pre-tool text → tool call event → tool result → post-tool text → final.
    assert text_items == ["Let me check.", "Done", "."]
    assert len(tool_calls) == 1 and tool_calls[0].tool_name == "echo"
    assert tool_calls[0].args == {"message": "hi"}
    assert tool_calls[0].tool_use_id == "tu_xyz"
    assert len(tool_results) == 1 and tool_results[0].success is True
    assert tool_results[0].result == {"echoed": "hi"}

    assert tool.calls == [{"message": "hi"}]

    res = finals[0]
    assert res.answer == "Let me check.Done."
    assert res.iterations == 2
    assert len(res.query_log.tool_calls) == 1
    call = res.query_log.tool_calls[0]
    assert call.tool_name == "echo"
    assert call.success is True
    assert call.result == {"echoed": "hi"}

    # Token accounting summed across both iterations.
    assert res.query_log.input_tokens == 50
    assert res.query_log.output_tokens == 12


def test_agent_propagates_tool_validation_error_to_claude() -> None:
    """A ToolValidationError must surface as is_error tool_result, not crash."""
    iter1 = FakeStream(
        tokens=[],
        usage=usage(),
        content=[tool_use_block("always_fails", {}, id_="bad_1")],
        stop_reason="tool_use",
    )
    iter2 = FakeStream(
        tokens=["I tried; it failed."],
        usage=usage(),
        content=[text_block("I tried; it failed.")],
        stop_reason="end_turn",
    )
    client = FakeClient([iter1, iter2])
    agent = Agent(tools=[_AlwaysFailsTool()], client=client)

    items = list(agent.stream("trigger", []))
    tool_results = [i for i in items if isinstance(i, ToolResultEvent)]
    finals = [i for i in items if isinstance(i, AgentResult)]

    assert len(tool_results) == 1
    assert tool_results[0].success is False
    assert tool_results[0].error == "nope"

    # The next request must include a tool_result with is_error=True.
    second_call = client.messages.calls[1]
    user_turn = second_call["messages"][-1]
    assert user_turn["role"] == "user"
    tool_result_block = user_turn["content"][0]
    assert tool_result_block["type"] == "tool_result"
    assert tool_result_block["tool_use_id"] == "bad_1"
    assert tool_result_block["is_error"] is True
    assert tool_result_block["content"] == "nope"

    # QueryLog must record the failed tool call.
    res = finals[0]
    assert len(res.query_log.tool_calls) == 1
    assert res.query_log.tool_calls[0].success is False
    assert res.query_log.tool_calls[0].error == "nope"


def test_agent_handles_unknown_tool_name() -> None:
    iter1 = FakeStream(
        tokens=[],
        usage=usage(),
        content=[tool_use_block("does_not_exist", {})],
        stop_reason="tool_use",
    )
    iter2 = FakeStream(
        tokens=["Sorry."],
        usage=usage(),
        content=[text_block("Sorry.")],
        stop_reason="end_turn",
    )
    client = FakeClient([iter1, iter2])
    agent = Agent(tools=[_EchoTool()], client=client)

    items = list(agent.stream("q", []))
    tool_results = [i for i in items if isinstance(i, ToolResultEvent)]
    assert tool_results[0].success is False
    assert "Unknown tool" in (tool_results[0].error or "")


# --- request shape -----------------------------------------------------------


def test_agent_sends_tools_payload_and_cached_system() -> None:
    client = FakeClient(
        [
            FakeStream(
                tokens=["ok"],
                usage=usage(),
                content=[text_block("ok")],
                stop_reason="end_turn",
            )
        ]
    )
    agent = Agent(tools=[_EchoTool()], client=client)

    list(agent.stream("q", []))

    kwargs = client.messages.calls[0]
    # System still has the cache breakpoint on the context block (from generation._build_system).
    system = kwargs["system"]
    assert system[1]["cache_control"] == {"type": "ephemeral"}
    # Tools payload included with cache breakpoint on the last tool.
    tools = kwargs["tools"]
    assert tools[-1]["cache_control"] == {"type": "ephemeral"}
    assert tools[0]["name"] == "echo"


def test_agent_omits_tools_when_none_registered() -> None:
    client = FakeClient(
        [
            FakeStream(
                tokens=["ok"],
                usage=usage(),
                content=[text_block("ok")],
                stop_reason="end_turn",
            )
        ]
    )
    agent = Agent(tools=[], client=client)
    list(agent.stream("q", []))
    assert "tools" not in client.messages.calls[0]


# --- safety ------------------------------------------------------------------


def test_agent_caps_iterations() -> None:
    """A model that always returns tool_use must stop at max_iterations."""

    def forever_stream() -> FakeStream:
        return FakeStream(
            tokens=[],
            usage=usage(),
            content=[tool_use_block("echo", {"message": "x"})],
            stop_reason="tool_use",
        )

    client = FakeClient([forever_stream() for _ in range(3)])
    agent = Agent(tools=[_EchoTool()], client=client, max_iterations=3)

    res = agent.run("loop", [])
    assert res.iterations == 3
    # Last tool_call recorded is the synthetic loop-cap entry.
    assert any(tc.tool_name == "<agent_loop>" for tc in res.query_log.tool_calls)


def test_agent_run_drains_stream() -> None:
    """`Agent.run` is the convenience non-streaming wrapper."""
    client = FakeClient(
        [
            FakeStream(
                tokens=["A", "B"],
                usage=usage(),
                content=[text_block("AB")],
                stop_reason="end_turn",
            )
        ]
    )
    agent = Agent(tools=[], client=client)
    res = agent.run("q", [])
    assert isinstance(res, AgentResult)
    assert res.answer == "AB"


def test_session_history_threaded_into_messages() -> None:
    from knowledge_rag.models import Session, Turn

    client = FakeClient(
        [
            FakeStream(
                tokens=["fine"],
                usage=usage(),
                content=[text_block("fine")],
                stop_reason="end_turn",
            )
        ]
    )
    agent = Agent(tools=[], client=client)
    sess = Session(
        history=[
            Turn(role="user", content="prior?"),
            Turn(role="assistant", content="prior answer."),
        ]
    )
    list(agent.stream("now?", [], session=sess))

    messages = client.messages.calls[0]["messages"]
    assert messages[0] == {"role": "user", "content": "prior?"}
    assert messages[1] == {"role": "assistant", "content": "prior answer."}
    assert messages[2] == {"role": "user", "content": "now?"}
