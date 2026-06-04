"""Best-effort live LangChain Google integration test."""

from __future__ import annotations

import json
import os
import warnings
from types import SimpleNamespace
from typing import Any

import pytest

from moduna_otel.services import moduna_langchain_callback_handler as module
from moduna_otel.services.moduna_langchain_callback_handler import (
    ModunaLangChainCallbackHandler,
)


class CollectedSpan:
    """Minimal span that stores attributes emitted during a live LangChain call."""

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
        """Return a fake context for optional debug logging."""
        return SimpleNamespace(trace_id=1, span_id=2)


class CollectedTracer:
    """Tracer fake that returns a single collected span."""

    def __init__(self, span: CollectedSpan) -> None:
        """Store the span returned by start_span."""
        self.span = span

    def start_span(self, name: str, kind: object) -> CollectedSpan:
        """Capture start arguments and return the span."""
        self.name = name
        self.kind = kind
        return self.span


def test_google_llm_collects_langchain_attributes_as_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Call Google through LangChain and print collected span attributes as JSON."""
    dotenv = pytest.importorskip("dotenv", reason="python-dotenv is unavailable")
    langchain_google = pytest.importorskip(
        "langchain_google_genai",
        reason="langchain-google-genai is unavailable",
    )
    dotenv.load_dotenv()

    api_key = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
    if not api_key or api_key is None:
        _warn_and_skip("GOOGLE_GENERATIVE_AI_API_KEY is unavailable")
        return

    span = CollectedSpan()
    monkeypatch.setattr(module.trace, "get_tracer", lambda _: CollectedTracer(span))
    handler = ModunaLangChainCallbackHandler(
        {"conversation_id": "google-live-conversation", "session_id": "google-live-session"}
    )
    model_name = os.getenv("GOOGLE_GENERATIVE_AI_MODEL", "gemini-2.5-flash")
    llm = _create_google_chat_model(
        langchain_google.ChatGoogleGenerativeAI,
        model_name,
        api_key,
    )

    try:
        llm.invoke(
            "Reply with exactly these two words: moduna ok",
            config={
                "callbacks": [handler],
                "metadata": {"test_name": "google-live-langchain"},
            },
        )
    except Exception as exc:  # pragma: no cover - depends on live provider availability.
        _warn_and_skip(f"Google Generative AI API is unavailable: {exc!r}")

    collected_json = json.dumps(span.attributes, sort_keys=True, indent=2, default=str)
    print(collected_json)

    assert span.ended is True
    assert span.attributes["gen_ai.system"] == "google"
    assert span.attributes["gen_ai.provider.name"] == "google"
    assert span.attributes["gen_ai.request.model"] == model_name
    assert "gen_ai.prompt.0.content" in span.attributes
    assert "gen_ai.completion.0.content" in span.attributes
    assert '"gen_ai.request.model"' in collected_json


def test_google_weather_tool_call_collects_attributes_as_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Call Google with a weather tool and print collected attributes as JSON."""
    dotenv = pytest.importorskip("dotenv", reason="python-dotenv is unavailable")
    langchain_google = pytest.importorskip(
        "langchain_google_genai",
        reason="langchain-google-genai is unavailable",
    )
    langchain_tools = pytest.importorskip(
        "langchain_core.tools",
        reason="langchain-core tools are unavailable",
    )
    dotenv.load_dotenv()

    api_key = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
    if not api_key or api_key is None:
        _warn_and_skip("GOOGLE_GENERATIVE_AI_API_KEY is unavailable")
        return

    span = CollectedSpan()
    monkeypatch.setattr(module.trace, "get_tracer", lambda _: CollectedTracer(span))
    handler = ModunaLangChainCallbackHandler(
        {
            "conversation_id": "google-weather-conversation",
            "session_id": "google-weather-session",
        }
    )
    model_name = os.getenv("GOOGLE_GENERATIVE_AI_MODEL", "gemini-2.5-flash")
    llm = _create_google_chat_model(
        langchain_google.ChatGoogleGenerativeAI,
        model_name,
        api_key,
        max_output_tokens=256,
    )

    @langchain_tools.tool
    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        return f"The weather in {location} is sunny and 72 F."

    try:
        response = llm.bind_tools([get_weather], tool_choice="get_weather").invoke(
            "Use the get_weather tool for San Francisco.",
            config={
                "callbacks": [handler],
                "metadata": {"test_name": "google-weather-tool-call"},
            },
        )
    except Exception as exc:  # pragma: no cover - depends on live provider availability.
        _warn_and_skip(f"Google Generative AI weather tool call is unavailable: {exc!r}")

    collected_json = json.dumps(span.attributes, sort_keys=True, indent=2, default=str)
    print(collected_json)

    tool_calls = getattr(response, "tool_calls", [])
    assert span.ended is True
    assert tool_calls
    assert tool_calls[0]["name"] == "get_weather"
    assert span.attributes["gen_ai.system"] == "google"
    assert span.attributes["gen_ai.provider.name"] == "google"
    assert span.attributes["gen_ai.request.model"] == model_name
    assert "gen_ai.tool.definitions" in span.attributes
    assert "tools" in span.attributes
    assert "get_weather" in collected_json


def _create_google_chat_model(
    model_class: type[Any],
    model_name: str,
    api_key: str,
    max_output_tokens: int = 32,
) -> Any:
    """Create ChatGoogleGenerativeAI across supported constructor variants."""
    common_params: dict[str, object] = {
        "model": model_name,
        "temperature": 0,
        "max_output_tokens": max_output_tokens,
    }
    try:
        return model_class(**common_params, google_api_key=api_key)
    except TypeError:
        return model_class(**common_params, api_key=api_key)


def _warn_and_skip(message: str) -> None:
    """Warn before skipping an unavailable live integration test."""
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    pytest.skip(message)
