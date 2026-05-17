"""AsyncAgent tests — mirror tests/test_agent.py with async semantics."""

from __future__ import annotations

from typing import Any

from knowledge_rag.agent import (
    AgentResult,
    AsyncAgent,
    ToolCallEvent,
    ToolResultEvent,
)
from knowledge_rag.tools.base import Tool, ToolValidationError
from tests._anthropic_fakes import (
    AsyncFakeClient,
    AsyncFakeStream,
    text_block,
    tool_use_block,
    usage,
)


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


async def _drain(agent_stream) -> list:
    return [item async for item in agent_stream]


async def test_async_agent_streams_text_when_no_tool_use() -> None:
    client = AsyncFakeClient(
        [
            AsyncFakeStream(
                tokens=("Hello", " ", "world"),
                usage=usage(),
                content=[text_block("Hello world")],
                stop_reason="end_turn",
            )
        ]
    )
    agent = AsyncAgent(tools=[_EchoTool()], client=client)

    items = await _drain(agent.stream("hi", []))
    text_items = [i for i in items if isinstance(i, str)]
    results = [i for i in items if isinstance(i, AgentResult)]

    assert text_items == ["Hello", " ", "world"]
    assert len(results) == 1
    assert results[0].answer == "Hello world"
    assert results[0].iterations == 1


async def test_async_agent_dispatches_tool_then_streams_final_answer() -> None:
    iter1 = AsyncFakeStream(
        tokens=("Let me check.",),
        usage=usage(input_tokens=20, output_tokens=8),
        content=[
            text_block("Let me check."),
            tool_use_block("echo", {"message": "hi"}, id_="tu_xyz"),
        ],
        stop_reason="tool_use",
    )
    iter2 = AsyncFakeStream(
        tokens=("Done", "."),
        usage=usage(input_tokens=30, output_tokens=4),
        content=[text_block("Done.")],
        stop_reason="end_turn",
    )
    tool = _EchoTool()
    client = AsyncFakeClient([iter1, iter2])
    agent = AsyncAgent(tools=[tool], client=client)

    items = await _drain(agent.stream("call echo", []))
    text_items = [i for i in items if isinstance(i, str)]
    tool_calls = [i for i in items if isinstance(i, ToolCallEvent)]
    tool_results = [i for i in items if isinstance(i, ToolResultEvent)]
    finals = [i for i in items if isinstance(i, AgentResult)]

    assert text_items == ["Let me check.", "Done", "."]
    assert tool_calls[0].tool_name == "echo"
    assert tool_results[0].success is True
    assert tool.calls == [{"message": "hi"}]
    assert finals[0].iterations == 2
    assert finals[0].query_log.input_tokens == 50


async def test_async_agent_propagates_validation_error() -> None:
    iter1 = AsyncFakeStream(
        tokens=(),
        usage=usage(),
        content=[tool_use_block("always_fails", {}, id_="bad_1")],
        stop_reason="tool_use",
    )
    iter2 = AsyncFakeStream(
        tokens=("recovered.",),
        usage=usage(),
        content=[text_block("recovered.")],
        stop_reason="end_turn",
    )
    client = AsyncFakeClient([iter1, iter2])
    agent = AsyncAgent(tools=[_AlwaysFailsTool()], client=client)

    items = await _drain(agent.stream("trigger", []))
    tool_results = [i for i in items if isinstance(i, ToolResultEvent)]
    assert tool_results[0].success is False
    assert tool_results[0].error == "nope"

    second = client.messages.calls[1]
    tr_block = second["messages"][-1]["content"][0]
    assert tr_block["is_error"] is True
    assert tr_block["content"] == "nope"


async def test_async_agent_caps_iterations() -> None:
    def forever() -> AsyncFakeStream:
        return AsyncFakeStream(
            tokens=(),
            usage=usage(),
            content=[tool_use_block("echo", {"message": "x"})],
            stop_reason="tool_use",
        )

    client = AsyncFakeClient([forever() for _ in range(3)])
    agent = AsyncAgent(tools=[_EchoTool()], client=client, max_iterations=3)

    res = await agent.run("loop", [])
    assert res.iterations == 3
    assert any(tc.tool_name == "<agent_loop>" for tc in res.query_log.tool_calls)


async def test_async_agent_run_returns_result() -> None:
    client = AsyncFakeClient(
        [
            AsyncFakeStream(
                tokens=("A", "B"),
                usage=usage(),
                content=[text_block("AB")],
                stop_reason="end_turn",
            )
        ]
    )
    agent = AsyncAgent(tools=[], client=client)
    res = await agent.run("q", [])
    assert isinstance(res, AgentResult)
    assert res.answer == "AB"


async def test_async_agent_omits_tools_when_empty() -> None:
    client = AsyncFakeClient(
        [
            AsyncFakeStream(
                tokens=("ok",),
                usage=usage(),
                content=[text_block("ok")],
                stop_reason="end_turn",
            )
        ]
    )
    agent = AsyncAgent(tools=[], client=client)
    await agent.run("q", [])
    assert "tools" not in client.messages.calls[0]
