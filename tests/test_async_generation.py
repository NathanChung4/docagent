"""AsyncGenerator tests — mirror tests/test_generation.py with async semantics."""

from __future__ import annotations

from knowledge_rag.generation import AsyncGenerator, GenerationConfig, GenerationResult
from knowledge_rag.models import Chunk, RetrievalResult, Session, SourceType, Turn
from tests._anthropic_fakes import AsyncFakeClient, AsyncFakeStream, text_block, usage


def _make_async_client(
    tokens: tuple[str, ...] = ("Hello", " ", "world"),
    *,
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    answer_text: str | None = None,
) -> AsyncFakeClient:
    text = answer_text if answer_text is not None else "".join(tokens)
    return AsyncFakeClient(
        [
            AsyncFakeStream(
                tokens=tokens,
                usage=usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_creation_input_tokens=cache_creation_input_tokens,
                    cache_read_input_tokens=cache_read_input_tokens,
                ),
                content=[text_block(text)],
            )
        ]
    )


def _result(chunk_id: str, content: str, *, title: str = "T", uri: str = "u") -> RetrievalResult:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="doc",
        content=content,
        source_type=SourceType.WIKI,
        title=title,
        uri=uri,
    )
    return RetrievalResult(chunk=chunk, score=0.5)


async def test_async_stream_yields_tokens_then_one_result() -> None:
    client = _make_async_client(tokens=("Hello", " ", "world"))
    gen = AsyncGenerator(client=client)

    items: list = []
    async for x in gen.stream("what?", []):
        items.append(x)

    text_items = [i for i in items if isinstance(i, str)]
    result_items = [i for i in items if isinstance(i, GenerationResult)]
    assert text_items == ["Hello", " ", "world"]
    assert len(result_items) == 1
    assert result_items[0].answer == "Hello world"


async def test_async_generate_returns_populated_query_log() -> None:
    client = _make_async_client(
        tokens=("A", "B"),
        input_tokens=42,
        output_tokens=7,
        cache_read_input_tokens=1000,
        cache_creation_input_tokens=2000,
    )
    gen = AsyncGenerator(client=client)
    out = await gen.generate("question?", [_result("c1", "x")])

    log = out.query_log
    assert log.query == "question?"
    assert log.answer == "AB"
    assert log.input_tokens == 42
    assert log.output_tokens == 7
    assert log.cached_tokens == 1000
    assert log.cache_creation_tokens == 2000
    assert log.first_token_ms > 0


async def test_async_session_history_threaded() -> None:
    client = _make_async_client()
    gen = AsyncGenerator(client=client)
    sess = Session(
        history=[
            Turn(role="user", content="prior?"),
            Turn(role="assistant", content="prior!"),
        ]
    )
    await gen.generate("now?", [], session=sess)

    kwargs = client.messages.calls[0]
    assert kwargs["messages"][-1] == {"role": "user", "content": "now?"}
    assert kwargs["messages"][0] == {"role": "user", "content": "prior?"}


async def test_async_uses_configured_model() -> None:
    client = _make_async_client()
    gen = AsyncGenerator(
        config=GenerationConfig(model="claude-test", max_tokens=99),
        client=client,
    )
    await gen.generate("q", [])
    kwargs = client.messages.calls[0]
    assert kwargs["model"] == "claude-test"
    assert kwargs["max_tokens"] == 99


async def test_async_system_payload_caches_context() -> None:
    client = _make_async_client()
    gen = AsyncGenerator(client=client)
    await gen.generate("q", [_result("c1", "body")])

    system = client.messages.calls[0]["system"]
    assert system[1]["cache_control"] == {"type": "ephemeral"}
    assert "body" in system[1]["text"]
