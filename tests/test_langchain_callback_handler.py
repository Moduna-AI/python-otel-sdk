"""Unit tests for Moduna's LangChain callback handler."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from moduna_otel.services import moduna_langchain_callback_handler as module
from moduna_otel.services.moduna_langchain_callback_handler import (
    ModunaLangChainCallbackHandler,
)


class FakeSpan:
    """Minimal span fake for LangChain handler assertions."""

    def __init__(self) -> None:
        """Create empty observation containers."""
        self.attributes: dict[str, object] = {}
        self.events: list[tuple[str, dict[str, object]]] = []
        self.exceptions: list[BaseException] = []
        self.ended = False

    def set_attribute(self, key: str, value: object) -> None:
        """Capture an attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, object]) -> None:
        """Capture an event."""
        self.events.append((name, attributes))

    def record_exception(self, exc: BaseException) -> None:
        """Capture an exception."""
        self.exceptions.append(exc)

    def set_status(self, status: object) -> None:
        """Capture span status."""
        self.status = status

    def end(self) -> None:
        """Mark the span ended."""
        self.ended = True

    def get_span_context(self) -> SimpleNamespace:
        """Return a fake context for debug logging."""
        return SimpleNamespace(trace_id=1, span_id=2)


class FakeTracer:
    """Tracer fake that returns a single span."""

    def __init__(self, span: FakeSpan) -> None:
        """Store the returned span."""
        self.span = span

    def start_span(self, name: str, kind: object) -> FakeSpan:
        """Capture start arguments and return the span."""
        self.name = name
        self.kind = kind
        return self.span


def test_llm_run_records_prompt_completion_usage_and_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM callbacks populate the core Moduna, LangChain, GenAI attributes."""
    span = FakeSpan()
    monkeypatch.setattr(module.trace, "get_tracer", Mock(return_value=FakeTracer(span)))
    handler = ModunaLangChainCallbackHandler({"conversation_id": "c1", "session_id": "s1"})

    handler.on_llm_start(
        {"id": ["langchain", "chat_models", "openai"], "name": "gpt-4o-mini"},
        ["hello"],
        run_id="run-1",
        tags=["prod"],
        invocation_params={"temperature": 0.2, "model": "gpt-4o-mini"},
    )
    handler.on_llm_new_token("hi", run_id="run-1")
    handler.on_llm_end(
        {
            "generations": [[{"text": "hi"}]],
            "llm_output": {"tokenUsage": {"promptTokens": 3, "totalTokens": 4}},
        },
        run_id="run-1",
    )

    assert span.attributes["moduna.framework"] == "langchain"
    assert span.attributes["moduna.conversation.id"] == "c1"
    assert span.attributes["moduna.session.id"] == "s1"
    assert span.attributes["gen_ai.prompt.0.content"] == "hello"
    assert span.attributes["gen_ai.completion.0.content"] == "hi"
    assert span.attributes["gen_ai.usage.prompt_tokens"] == 3
    assert span.attributes["gen_ai.usage.completion_tokens"] == 1
    assert span.attributes["gen_ai.usage.total_tokens"] == 4
    assert span.ended is True
    assert "run-1" not in handler.runs


def test_chat_model_metadata_overrides_default_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """LangChain metadata can override default conversation and session IDs."""
    span = FakeSpan()
    monkeypatch.setattr(module.trace, "get_tracer", Mock(return_value=FakeTracer(span)))
    handler = ModunaLangChainCallbackHandler({"conversation_id": "default"})
    message = SimpleNamespace(content="hello", type="human")

    handler.on_chat_model_start(
        {"id": ["langchain", "chat_models", "anthropic"]},
        [[message]],
        run_id="run-2",
        metadata={"conversationId": "override", "moduna.session.id": "s2"},
    )

    assert span.attributes["moduna.conversation.id"] == "override"
    assert span.attributes["moduna.session.id"] == "s2"
    assert span.attributes["gen_ai.operation.name"] == "chat"


def test_llm_error_records_exception_and_ends_span(monkeypatch: pytest.MonkeyPatch) -> None:
    """LangChain error callback records the failure and closes the run span."""
    span = FakeSpan()
    monkeypatch.setattr(module.trace, "get_tracer", Mock(return_value=FakeTracer(span)))
    handler = ModunaLangChainCallbackHandler()
    error = RuntimeError("failed")

    handler.on_llm_start({"id": ["x"]}, ["hello"], run_id="run-3")
    handler.on_llm_error(error, run_id="run-3")

    assert span.exceptions == [error]
    assert span.ended is True
    assert "run-3" not in handler.runs
