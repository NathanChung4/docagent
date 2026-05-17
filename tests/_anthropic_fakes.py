"""Anthropic SDK fakes shared by Generator and Agent tests.

Each call to `client.messages.stream(...)` returns the next pre-canned
FakeStream (popped FIFO). Single-iteration tests pass a one-element list;
multi-iteration tests (the agent loop) pass several.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any


class FakeStream:
    """Stand-in for `client.messages.stream(...).__enter__()` return value."""

    def __init__(
        self,
        tokens: Sequence[str],
        usage: Any,
        content: Sequence[Any],
        stop_reason: str = "end_turn",
    ) -> None:
        self._tokens = list(tokens)
        self._usage = usage
        self._content = list(content)
        self._stop_reason = stop_reason

    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    @property
    def text_stream(self):
        return iter(self._tokens)

    def get_final_message(self) -> Any:
        return SimpleNamespace(
            usage=self._usage,
            content=self._content,
            stop_reason=self._stop_reason,
        )


class FakeMessages:
    def __init__(self, scripted: Sequence[FakeStream]) -> None:
        self._scripted = list(scripted)
        self.calls: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> FakeStream:
        self.calls.append(kwargs)
        if not self._scripted:
            raise AssertionError("Caller issued more requests than the script provides.")
        return self._scripted.pop(0)

    @property
    def last_kwargs(self) -> dict[str, Any] | None:
        return self.calls[-1] if self.calls else None


class FakeClient:
    def __init__(self, scripted: Sequence[FakeStream]) -> None:
        self.messages = FakeMessages(scripted)


def usage(
    input_tokens: int = 10,
    output_tokens: int = 5,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> Any:
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
    )


def text_block(s: str) -> Any:
    return SimpleNamespace(type="text", text=s)


def tool_use_block(name: str, args: dict[str, Any], id_: str = "tu_1") -> Any:
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=args)


# --- async variants ---------------------------------------------------------


class AsyncFakeStream:
    """Async stand-in for `client.messages.stream(...).__aenter__()` return value."""

    def __init__(
        self,
        tokens: Sequence[str],
        usage: Any,
        content: Sequence[Any],
        stop_reason: str = "end_turn",
    ) -> None:
        self._tokens = list(tokens)
        self._usage = usage
        self._content = list(content)
        self._stop_reason = stop_reason

    async def __aenter__(self) -> AsyncFakeStream:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    @property
    def text_stream(self):
        async def _gen():
            for t in self._tokens:
                yield t

        return _gen()

    async def get_final_message(self) -> Any:
        return SimpleNamespace(
            usage=self._usage,
            content=self._content,
            stop_reason=self._stop_reason,
        )


class AsyncFakeMessages:
    def __init__(self, scripted: Sequence[AsyncFakeStream]) -> None:
        self._scripted = list(scripted)
        self.calls: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> AsyncFakeStream:
        self.calls.append(kwargs)
        if not self._scripted:
            raise AssertionError("Caller issued more requests than the script provides.")
        return self._scripted.pop(0)


class AsyncFakeClient:
    def __init__(self, scripted: Sequence[AsyncFakeStream]) -> None:
        self.messages = AsyncFakeMessages(scripted)
