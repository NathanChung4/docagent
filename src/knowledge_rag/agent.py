"""Agent loop: lets Claude decide between answering directly and calling a tool.

Built on the same streaming primitives as Generator, but adds:
  - tools=[...] in the request, populated from a Domain's tool registry
  - a loop that dispatches tool_use blocks back to the right Tool.run()
  - tool_result wiring so failures (ToolValidationError) flow back as
    is_error=True blocks the model can self-correct on
  - ToolCall records appended to the QueryLog for observability

Streaming preserved end-to-end: text tokens are yielded as strings, tool
events are yielded as typed dataclasses, and exactly one AgentResult is
yielded at the end.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from knowledge_rag.generation import (
    GenerationConfig,
    build_messages,
    build_system,
    compute_cost,
    format_context,
)
from knowledge_rag.models import QueryLog, RetrievalResult, Session, ToolCall
from knowledge_rag.tools.base import Tool, ToolValidationError

DEFAULT_MAX_ITERATIONS = 5


@dataclass
class ToolCallEvent:
    """Emitted before a tool runs so UIs can show 'Calling tool: X...'."""

    tool_name: str
    args: dict[str, Any]
    tool_use_id: str


@dataclass
class ToolResultEvent:
    """Emitted after a tool runs (success or validation failure)."""

    tool_name: str
    tool_use_id: str
    result: dict[str, Any] | None
    success: bool
    error: str | None = None


@dataclass
class AgentResult:
    """Final agent output. `query_log.tool_calls` lists every dispatch."""

    answer: str
    query_log: QueryLog
    iterations: int = 1


@dataclass
class _UsageAccumulator:
    """Sum token counts across iterations of the agent loop."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

    def add(self, usage: Any) -> None:
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.cache_creation_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0


def _block_to_dict(block: Any) -> dict[str, Any]:
    """Convert one assistant content block (SDK or SimpleNamespace) to a dict.

    Round-trip is important: the dict goes back to Claude as part of the
    assistant turn on the next iteration, so it must round-trip cleanly.
    """
    btype = block.type
    if btype == "text":
        return {"type": "text", "text": block.text}
    if btype == "tool_use":
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    raise ValueError(f"Unexpected assistant content block type: {btype!r}")


def _serialize_tool_result(result: dict[str, Any]) -> str:
    """Tool results go back to Claude as a string. JSON keeps structure visible."""
    return json.dumps(result, default=str)


def _build_tools_payload(tools: Sequence[Tool]) -> list[dict[str, Any]]:
    """Render tools and put a cache breakpoint on the last one.

    Caching the tail caches the system + tool prefix together (Anthropic
    walks cache_control markers in order; the last one absorbs everything
    above it). Cuts the per-query cost when the same tool set repeats.
    """
    schemas = [tool.to_anthropic_schema() for tool in tools]
    if schemas:
        schemas[-1] = {**schemas[-1], "cache_control": {"type": "ephemeral"}}
    return schemas


class Agent:
    """Claude wrapper that supports tool use on top of grounded RAG context."""

    def __init__(
        self,
        tools: Sequence[Tool],
        config: GenerationConfig | None = None,
        client: Any = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> None:
        self.tools = list(tools)
        self.tools_by_name = {t.name: t for t in self.tools}
        self.config = config or GenerationConfig()
        self._client = client
        self.max_iterations = max_iterations
        self._tools_payload = _build_tools_payload(self.tools)

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def _dispatch(
        self,
        block: Any,
    ) -> tuple[dict[str, Any], ToolResultEvent, ToolCall]:
        """Run one tool_use block. Returns the tool_result block + events."""
        tool = self.tools_by_name.get(block.name)
        args = dict(block.input or {})
        if tool is None:
            err = (
                f"Unknown tool '{block.name}'. Available: "
                f"{', '.join(sorted(self.tools_by_name)) or '(none)'}."
            )
            tr_block = {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": err,
                "is_error": True,
            }
            event = ToolResultEvent(
                tool_name=block.name,
                tool_use_id=block.id,
                result=None,
                success=False,
                error=err,
            )
            call = ToolCall(tool_name=block.name, args=args, error=err, success=False)
            return tr_block, event, call

        try:
            result = tool.run(**args)
            tr_block = {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": _serialize_tool_result(result),
            }
            event = ToolResultEvent(
                tool_name=block.name,
                tool_use_id=block.id,
                result=result,
                success=True,
            )
            call = ToolCall(tool_name=block.name, args=args, result=result, success=True)
            return tr_block, event, call
        except ToolValidationError as e:
            err = str(e)
            tr_block = {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": err,
                "is_error": True,
            }
            event = ToolResultEvent(
                tool_name=block.name,
                tool_use_id=block.id,
                result=None,
                success=False,
                error=err,
            )
            call = ToolCall(tool_name=block.name, args=args, error=err, success=False)
            return tr_block, event, call

    def stream(
        self,
        query: str,
        context: Sequence[RetrievalResult] = (),
        session: Session | None = None,
    ) -> Iterator[str | ToolCallEvent | ToolResultEvent | AgentResult]:
        """Stream tokens + tool events; yield exactly one AgentResult at the end."""
        client = self._get_client()
        system = build_system(self.config.system_prompt, format_context(context))
        messages = build_messages(query, session)

        usage_acc = _UsageAccumulator()
        tool_calls_log: list[ToolCall] = []
        answer_chunks: list[str] = []

        start = time.perf_counter()
        first_token_ms: float | None = None
        iterations = 0

        for iteration in range(self.max_iterations):
            iterations = iteration + 1
            request_kwargs: dict[str, Any] = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "system": system,
                "messages": messages,
            }
            if self._tools_payload:
                request_kwargs["tools"] = self._tools_payload

            with client.messages.stream(**request_kwargs) as stream:
                for text in stream.text_stream:
                    if first_token_ms is None and text:
                        first_token_ms = (time.perf_counter() - start) * 1000.0
                    answer_chunks.append(text)
                    yield text
                final_message = stream.get_final_message()

            usage_acc.add(final_message.usage)

            stop_reason = getattr(final_message, "stop_reason", None)
            if stop_reason != "tool_use":
                break

            # Dispatch every tool_use block in the assistant turn.
            tool_result_blocks: list[dict[str, Any]] = []
            for block in final_message.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                yield ToolCallEvent(
                    tool_name=block.name,
                    args=dict(block.input or {}),
                    tool_use_id=block.id,
                )
                tr_block, event, call = self._dispatch(block)
                tool_result_blocks.append(tr_block)
                tool_calls_log.append(call)
                yield event

            messages.append(
                {"role": "assistant", "content": [_block_to_dict(b) for b in final_message.content]}
            )
            messages.append({"role": "user", "content": tool_result_blocks})
        else:
            # Loop fell through max_iterations without the model finishing.
            # Surface as a tool call with error so it's visible in QueryLog.
            tool_calls_log.append(
                ToolCall(
                    tool_name="<agent_loop>",
                    args={"max_iterations": self.max_iterations},
                    error=f"Agent did not finish within {self.max_iterations} iterations.",
                    success=False,
                )
            )

        latency_ms = (time.perf_counter() - start) * 1000.0
        answer = "".join(answer_chunks)

        log = QueryLog(
            query=query,
            answer=answer,
            session_id=session.session_id if session is not None else None,
            retrieved_chunk_ids=[r.chunk.chunk_id for r in context],
            tool_calls=tool_calls_log,
            input_tokens=usage_acc.input_tokens,
            output_tokens=usage_acc.output_tokens,
            cached_tokens=usage_acc.cache_read_tokens,
            cache_creation_tokens=usage_acc.cache_creation_tokens,
            cost_usd=compute_cost(
                usage_acc.input_tokens,
                usage_acc.output_tokens,
                usage_acc.cache_creation_tokens,
                usage_acc.cache_read_tokens,
            ),
            latency_ms=latency_ms,
            first_token_ms=first_token_ms or 0.0,
        )
        yield AgentResult(answer=answer, query_log=log, iterations=iterations)

    def run(
        self,
        query: str,
        context: Sequence[RetrievalResult] = (),
        session: Session | None = None,
    ) -> AgentResult:
        """Non-streaming convenience wrapper. Drains the stream, returns the result."""
        result: AgentResult | None = None
        for item in self.stream(query, context, session):
            if isinstance(item, AgentResult):
                result = item
        assert result is not None, "stream must yield an AgentResult"
        return result


# --- async agent -------------------------------------------------------------


class AsyncAgent:
    """Async-native agent for use under FastAPI. Tool dispatch runs in a thread
    so a slow tool doesn't block the event loop.

    Control flow and emitted events match `Agent` — when changing one, change
    both. The duplication is deliberate: the SDK's sync and async stream
    contexts can't be unified without losing type safety.
    """

    def __init__(
        self,
        tools: Sequence[Tool],
        config: GenerationConfig | None = None,
        client: Any = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> None:
        self.tools = list(tools)
        self.tools_by_name = {t.name: t for t in self.tools}
        self.config = config or GenerationConfig()
        self._client = client
        self.max_iterations = max_iterations
        self._tools_payload = _build_tools_payload(self.tools)

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic()
        return self._client

    async def _dispatch(
        self,
        block: Any,
    ) -> tuple[dict[str, Any], ToolResultEvent, ToolCall]:
        tool = self.tools_by_name.get(block.name)
        args = dict(block.input or {})
        if tool is None:
            err = (
                f"Unknown tool '{block.name}'. Available: "
                f"{', '.join(sorted(self.tools_by_name)) or '(none)'}."
            )
            tr_block = {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": err,
                "is_error": True,
            }
            return (
                tr_block,
                ToolResultEvent(
                    tool_name=block.name,
                    tool_use_id=block.id,
                    result=None,
                    success=False,
                    error=err,
                ),
                ToolCall(tool_name=block.name, args=args, error=err, success=False),
            )

        try:
            # tools are sync today; off-thread so a slow xlsx parse doesn't
            # stall other concurrent requests sharing the event loop.
            result = await asyncio.to_thread(lambda: tool.run(**args))
            return (
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": _serialize_tool_result(result),
                },
                ToolResultEvent(
                    tool_name=block.name,
                    tool_use_id=block.id,
                    result=result,
                    success=True,
                ),
                ToolCall(tool_name=block.name, args=args, result=result, success=True),
            )
        except ToolValidationError as e:
            err = str(e)
            return (
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": err,
                    "is_error": True,
                },
                ToolResultEvent(
                    tool_name=block.name,
                    tool_use_id=block.id,
                    result=None,
                    success=False,
                    error=err,
                ),
                ToolCall(tool_name=block.name, args=args, error=err, success=False),
            )

    async def stream(
        self,
        query: str,
        context: Sequence[RetrievalResult] = (),
        session: Session | None = None,
    ) -> AsyncIterator[str | ToolCallEvent | ToolResultEvent | AgentResult]:
        client = self._get_client()
        system = build_system(self.config.system_prompt, format_context(context))
        messages = build_messages(query, session)

        usage_acc = _UsageAccumulator()
        tool_calls_log: list[ToolCall] = []
        answer_chunks: list[str] = []

        start = time.perf_counter()
        first_token_ms: float | None = None
        iterations = 0

        for iteration in range(self.max_iterations):
            iterations = iteration + 1
            request_kwargs: dict[str, Any] = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "system": system,
                "messages": messages,
            }
            if self._tools_payload:
                request_kwargs["tools"] = self._tools_payload

            async with client.messages.stream(**request_kwargs) as stream:
                async for text in stream.text_stream:
                    if first_token_ms is None and text:
                        first_token_ms = (time.perf_counter() - start) * 1000.0
                    answer_chunks.append(text)
                    yield text
                final_message = await stream.get_final_message()

            usage_acc.add(final_message.usage)

            if getattr(final_message, "stop_reason", None) != "tool_use":
                break

            tool_result_blocks: list[dict[str, Any]] = []
            for block in final_message.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                yield ToolCallEvent(
                    tool_name=block.name,
                    args=dict(block.input or {}),
                    tool_use_id=block.id,
                )
                tr_block, event, call = await self._dispatch(block)
                tool_result_blocks.append(tr_block)
                tool_calls_log.append(call)
                yield event

            messages.append(
                {"role": "assistant", "content": [_block_to_dict(b) for b in final_message.content]}
            )
            messages.append({"role": "user", "content": tool_result_blocks})
        else:
            tool_calls_log.append(
                ToolCall(
                    tool_name="<agent_loop>",
                    args={"max_iterations": self.max_iterations},
                    error=f"Agent did not finish within {self.max_iterations} iterations.",
                    success=False,
                )
            )

        latency_ms = (time.perf_counter() - start) * 1000.0
        answer = "".join(answer_chunks)

        log = QueryLog(
            query=query,
            answer=answer,
            session_id=session.session_id if session is not None else None,
            retrieved_chunk_ids=[r.chunk.chunk_id for r in context],
            tool_calls=tool_calls_log,
            input_tokens=usage_acc.input_tokens,
            output_tokens=usage_acc.output_tokens,
            cached_tokens=usage_acc.cache_read_tokens,
            cache_creation_tokens=usage_acc.cache_creation_tokens,
            cost_usd=compute_cost(
                usage_acc.input_tokens,
                usage_acc.output_tokens,
                usage_acc.cache_creation_tokens,
                usage_acc.cache_read_tokens,
            ),
            latency_ms=latency_ms,
            first_token_ms=first_token_ms or 0.0,
        )
        yield AgentResult(answer=answer, query_log=log, iterations=iterations)

    async def run(
        self,
        query: str,
        context: Sequence[RetrievalResult] = (),
        session: Session | None = None,
    ) -> AgentResult:
        """Non-streaming convenience wrapper. Drains the stream, returns the result."""
        result: AgentResult | None = None
        async for item in self.stream(query, context, session):
            if isinstance(item, AgentResult):
                result = item
        assert result is not None, "stream must yield an AgentResult"
        return result
