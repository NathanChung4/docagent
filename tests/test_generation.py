"""Tests for the Claude generation pipeline.

The Anthropic SDK is mocked end-to-end via the shared fakes in
`tests/_anthropic_fakes.py` so these run in CI without ANTHROPIC_API_KEY and
without network.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from knowledge_rag.generation import (
    CACHE_READ_COST_PER_M,
    CACHE_WRITE_COST_PER_M,
    INPUT_COST_PER_M,
    OUTPUT_COST_PER_M,
    GenerationConfig,
    GenerationResult,
    Generator,
    compute_cost,
    format_context,
)
from knowledge_rag.models import Chunk, RetrievalResult, Session, SourceType, Turn
from tests._anthropic_fakes import (
    FakeClient,
    FakeStream,
    text_block,
    usage,
)


def _make_client(
    tokens: Sequence[str] = ("Hello", " ", "world"),
    *,
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    answer_text: str | None = None,
) -> FakeClient:
    text = answer_text if answer_text is not None else "".join(tokens)
    return FakeClient(
        [
            FakeStream(
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


# --- helpers -----------------------------------------------------------------


def test_format_context_numbers_chunks() -> None:
    rendered = format_context(
        [
            _result("a", "alpha body", title="Alpha", uri="/a.md"),
            _result("b", "beta body", title="Beta", uri="/b.md"),
        ]
    )
    assert "[1] Alpha" in rendered
    assert "/a.md" in rendered
    assert "alpha body" in rendered
    assert "[2] Beta" in rendered
    assert "beta body" in rendered


def test_format_context_handles_empty() -> None:
    assert "no context" in format_context([])


def test_compute_cost_matches_unit_pricing() -> None:
    cost = compute_cost(
        input_tokens=1_000_000,
        output_tokens=0,
        cache_write_tokens=0,
        cache_read_tokens=0,
    )
    assert cost == pytest.approx(INPUT_COST_PER_M)

    cost = compute_cost(
        input_tokens=0,
        output_tokens=1_000_000,
        cache_write_tokens=1_000_000,
        cache_read_tokens=1_000_000,
    )
    expected = OUTPUT_COST_PER_M + CACHE_WRITE_COST_PER_M + CACHE_READ_COST_PER_M
    assert cost == pytest.approx(expected)


# --- streaming behavior ------------------------------------------------------


def test_stream_yields_tokens_then_one_result() -> None:
    client = _make_client(tokens=["Hello", " ", "world"])
    gen = Generator(client=client)

    items = list(gen.stream("what?", []))
    text_items = [i for i in items if isinstance(i, str)]
    result_items = [i for i in items if isinstance(i, GenerationResult)]

    assert text_items == ["Hello", " ", "world"]
    assert len(result_items) == 1
    assert result_items[0].answer == "Hello world"


def test_generate_returns_populated_query_log() -> None:
    client = _make_client(
        tokens=["A", "B"],
        input_tokens=42,
        output_tokens=7,
        cache_read_input_tokens=1000,
        cache_creation_input_tokens=2000,
    )
    gen = Generator(client=client)
    chunks = [_result("c1", "x"), _result("c2", "y")]

    out = gen.generate("question?", chunks)

    assert isinstance(out, GenerationResult)
    log = out.query_log
    assert log.query == "question?"
    assert log.answer == "AB"
    assert log.retrieved_chunk_ids == ["c1", "c2"]
    assert log.input_tokens == 42
    assert log.output_tokens == 7
    assert log.cached_tokens == 1000
    assert log.cache_creation_tokens == 2000
    expected_cost = compute_cost(42, 7, 2000, 1000)
    assert log.cost_usd == pytest.approx(expected_cost)
    assert log.latency_ms >= 0.0
    assert log.first_token_ms > 0.0


def test_generate_handles_empty_context() -> None:
    client = _make_client(tokens=["I don't know."])
    gen = Generator(client=client)
    out = gen.generate("???", [])
    assert "I don't know" in out.answer
    assert out.query_log.retrieved_chunk_ids == []


# --- prompt construction (verified through the kwargs the client received) ---


def test_system_payload_has_cache_control_on_context_block() -> None:
    client = _make_client()
    gen = Generator(client=client)
    list(gen.stream("q", [_result("c1", "body content")]))

    kwargs = client.messages.last_kwargs
    assert kwargs is not None
    system = kwargs["system"]
    assert len(system) == 2
    assert "cache_control" not in system[0]
    assert system[1]["cache_control"] == {"type": "ephemeral"}
    assert "[1]" in system[1]["text"]
    assert "body content" in system[1]["text"]


def test_user_query_goes_in_messages_uncached() -> None:
    client = _make_client()
    gen = Generator(client=client)
    list(gen.stream("explain feature_x", []))

    kwargs = client.messages.last_kwargs
    assert kwargs["messages"] == [{"role": "user", "content": "explain feature_x"}]


def test_session_history_prepended_to_messages() -> None:
    client = _make_client()
    gen = Generator(client=client)
    sess = Session(
        history=[
            Turn(role="user", content="what is feature_x?"),
            Turn(role="assistant", content="feature_x is a configurable component."),
        ]
    )
    list(gen.stream("what about its --thresh param?", [], session=sess))

    kwargs = client.messages.last_kwargs
    assert kwargs["messages"] == [
        {"role": "user", "content": "what is feature_x?"},
        {"role": "assistant", "content": "feature_x is a configurable component."},
        {"role": "user", "content": "what about its --thresh param?"},
    ]


def test_session_id_recorded_on_query_log() -> None:
    client = _make_client()
    gen = Generator(client=client)
    sess = Session()
    out = gen.generate("q", [], session=sess)
    assert out.query_log.session_id == sess.session_id


def test_uses_configured_model_and_max_tokens() -> None:
    client = _make_client()
    gen = Generator(
        config=GenerationConfig(model="claude-test", max_tokens=123),
        client=client,
    )
    list(gen.stream("q", []))
    kwargs = client.messages.last_kwargs
    assert kwargs["model"] == "claude-test"
    assert kwargs["max_tokens"] == 123


def test_default_config_targets_opus_4_7() -> None:
    """Spec.md commits to Opus for hardware-validation Q&A — guard against silent downgrade."""
    cfg = GenerationConfig()
    assert cfg.model == "claude-opus-4-7"
