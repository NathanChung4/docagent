"""Claude-powered answer generation: streaming, prompt caching, source attribution.

Caching: system prompt + numbered context block sit in `system=` with one
ephemeral breakpoint on the context block; user query and prior session turns
go in `messages` so the prefix stays cached across requests. Below Opus's
~4K-token minimum, the breakpoint is silently ignored.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from knowledge_rag.models import QueryLog, RetrievalResult, Session

DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_MAX_TOKENS = 4096

# Opus 4.7 pricing in USD per 1M tokens (cached: 2026-04-15).
INPUT_COST_PER_M = 5.00
OUTPUT_COST_PER_M = 25.00
CACHE_WRITE_COST_PER_M = 6.25
CACHE_READ_COST_PER_M = 0.50

SYSTEM_PROMPT = """You are an expert technical assistant for a hardware-validation engineering team. \
Answer questions using ONLY the numbered context provided below. Each context block is labeled [1], [2], etc.

Rules:
1. Cite the context blocks you used by appending [N] inline at the end of each claim that depends \
on them. Multiple sources: [1][3].
2. If the context does not contain enough information to answer, say "I don't know based on the \
available documents." Do not invent facts, parameter names, register names, or values.
3. Be concise. Prefer code blocks for code, lists for enumerations, plain prose for explanation.
4. If the question is ambiguous, answer the most likely interpretation, then briefly note the \
ambiguity at the end."""


@dataclass
class GenerationConfig:
    """Tunables for one Generator instance. Defaults match spec.md targets."""

    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    system_prompt: str = SYSTEM_PROMPT


@dataclass
class GenerationResult:
    """Final result of a generate/stream call. `query_log` is fully populated."""

    answer: str
    query_log: QueryLog


# --- prompt construction -----------------------------------------------------


def format_context(results: Sequence[RetrievalResult]) -> str:
    """Render retrieval hits as a numbered context block for the prompt."""
    if not results:
        return "(no context retrieved)"
    parts: list[str] = []
    for i, r in enumerate(results, start=1):
        chunk = r.chunk
        header = f"[{i}] {chunk.title} ({chunk.source_type.value}) — {chunk.uri}"
        parts.append(f"{header}\n{chunk.content}")
    return "\n\n".join(parts)


def build_system(system_prompt: str, context_block: str) -> list[dict[str, Any]]:
    """System content blocks. Cache breakpoint on the (large) context block."""
    return [
        {"type": "text", "text": system_prompt},
        {
            "type": "text",
            "text": f"\n\nContext documents:\n\n{context_block}",
            "cache_control": {"type": "ephemeral"},
        },
    ]


def build_messages(query: str, session: Session | None) -> list[dict[str, Any]]:
    """Convert prior session turns + current query into the messages array."""
    messages: list[dict[str, Any]] = []
    if session is not None:
        for turn in session.history:
            messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": query})
    return messages


# --- cost ---------------------------------------------------------------------


def compute_cost(
    input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int,
    cache_read_tokens: int,
) -> float:
    """Return the dollar cost of one request from token counts."""
    return (
        input_tokens * INPUT_COST_PER_M
        + output_tokens * OUTPUT_COST_PER_M
        + cache_write_tokens * CACHE_WRITE_COST_PER_M
        + cache_read_tokens * CACHE_READ_COST_PER_M
    ) / 1_000_000


# --- generator ---------------------------------------------------------------


class Generator:
    """Streaming Claude wrapper. Produces grounded answers from retrieved context."""

    def __init__(
        self,
        config: GenerationConfig | None = None,
        client: Any = None,
    ) -> None:
        self.config = config or GenerationConfig()
        self._client = client

    def _get_client(self) -> Any:
        """Lazy-construct the Anthropic client. Reads ANTHROPIC_API_KEY from env."""
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def stream(
        self,
        query: str,
        context: Sequence[RetrievalResult],
        session: Session | None = None,
    ) -> Iterator[str | GenerationResult]:
        """Stream tokens, then yield exactly one final GenerationResult."""
        client = self._get_client()
        context_block = format_context(context)
        system = build_system(self.config.system_prompt, context_block)
        messages = build_messages(query, session)

        start = time.perf_counter()
        first_token_ms: float | None = None

        with client.messages.stream(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=system,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                if first_token_ms is None and text:
                    first_token_ms = (time.perf_counter() - start) * 1000.0
                yield text
            final_message = stream.get_final_message()

        latency_ms = (time.perf_counter() - start) * 1000.0

        usage = final_message.usage
        answer = "".join(block.text for block in final_message.content if block.type == "text")

        log = QueryLog(
            query=query,
            answer=answer,
            session_id=session.session_id if session is not None else None,
            retrieved_chunk_ids=[r.chunk.chunk_id for r in context],
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_tokens=usage.cache_read_input_tokens,
            cache_creation_tokens=usage.cache_creation_input_tokens,
            cost_usd=compute_cost(
                usage.input_tokens,
                usage.output_tokens,
                usage.cache_creation_input_tokens,
                usage.cache_read_input_tokens,
            ),
            latency_ms=latency_ms,
            first_token_ms=first_token_ms or 0.0,
        )
        yield GenerationResult(answer=answer, query_log=log)

    def generate(
        self,
        query: str,
        context: Sequence[RetrievalResult],
        session: Session | None = None,
    ) -> GenerationResult:
        """Non-streaming convenience wrapper. Drains the stream, returns the result."""
        result: GenerationResult | None = None
        for item in self.stream(query, context, session):
            if isinstance(item, GenerationResult):
                result = item
        assert result is not None, "stream must yield a GenerationResult"
        return result


# --- async generator ---------------------------------------------------------


class AsyncGenerator:
    """Async-native streaming wrapper for use under FastAPI / SSE.

    Mirrors `Generator` but uses `anthropic.AsyncAnthropic`. Shares all prompt
    construction and cost helpers with the sync version — only the I/O is
    different.
    """

    def __init__(
        self,
        config: GenerationConfig | None = None,
        client: Any = None,
    ) -> None:
        self.config = config or GenerationConfig()
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic()
        return self._client

    async def stream(
        self,
        query: str,
        context: Sequence[RetrievalResult],
        session: Session | None = None,
    ) -> AsyncIterator[str | GenerationResult]:
        """Stream tokens, then yield exactly one final GenerationResult."""
        client = self._get_client()
        context_block = format_context(context)
        system = build_system(self.config.system_prompt, context_block)
        messages = build_messages(query, session)

        start = time.perf_counter()
        first_token_ms: float | None = None

        async with client.messages.stream(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=system,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                if first_token_ms is None and text:
                    first_token_ms = (time.perf_counter() - start) * 1000.0
                yield text
            final_message = await stream.get_final_message()

        latency_ms = (time.perf_counter() - start) * 1000.0
        usage = final_message.usage
        answer = "".join(block.text for block in final_message.content if block.type == "text")

        log = QueryLog(
            query=query,
            answer=answer,
            session_id=session.session_id if session is not None else None,
            retrieved_chunk_ids=[r.chunk.chunk_id for r in context],
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_tokens=usage.cache_read_input_tokens,
            cache_creation_tokens=usage.cache_creation_input_tokens,
            cost_usd=compute_cost(
                usage.input_tokens,
                usage.output_tokens,
                usage.cache_creation_input_tokens,
                usage.cache_read_input_tokens,
            ),
            latency_ms=latency_ms,
            first_token_ms=first_token_ms or 0.0,
        )
        yield GenerationResult(answer=answer, query_log=log)

    async def generate(
        self,
        query: str,
        context: Sequence[RetrievalResult],
        session: Session | None = None,
    ) -> GenerationResult:
        """Non-streaming convenience wrapper. Drains the stream, returns the result."""
        result: GenerationResult | None = None
        async for item in self.stream(query, context, session):
            if isinstance(item, GenerationResult):
                result = item
        assert result is not None, "stream must yield a GenerationResult"
        return result
