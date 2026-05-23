"""Unit tests for the ModunaOTEL public service."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from moduna_otel import ModunaOTEL
from moduna_otel.services import moduna_otel as module


class FakeProvider:
    """Small provider fake that captures processors and shutdown calls."""

    created = 0

    def __init__(self, resource: object) -> None:
        """Store construction inputs for assertions."""
        FakeProvider.created += 1
        self.resource = resource
        self.processors: list[object] = []
        self.shutdown_calls = 0

    def add_span_processor(self, processor: object) -> None:
        """Capture configured span processors."""
        self.processors.append(processor)

    def shutdown(self) -> None:
        """Record shutdown calls."""
        self.shutdown_calls += 1


class FakeSpan:
    """Minimal span fake used by instrumentation tests."""

    def __init__(self) -> None:
        """Create empty span observation state."""
        self.attributes: dict[str, object] = {}
        self.exceptions: list[BaseException] = []
        self.ended = False

    def set_attribute(self, key: str, value: object) -> None:
        """Capture span attributes."""
        self.attributes[key] = value

    def record_exception(self, exc: BaseException) -> None:
        """Capture recorded exceptions."""
        self.exceptions.append(exc)

    def set_status(self, status: object) -> None:
        """Capture status updates."""
        self.status = status

    def end(self) -> None:
        """Mark the span ended."""
        self.ended = True


class FakeTracer:
    """Tracer fake that returns a known span."""

    def __init__(self, span: FakeSpan) -> None:
        """Store the span returned by start_span."""
        self.span = span

    def start_span(self, name: str, kind: object) -> FakeSpan:
        """Return the configured span."""
        self.name = name
        self.kind = kind
        return self.span


def test_constructor_initializes_singleton_provider_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multiple ModunaOTEL instances share one provider/exporter setup."""
    FakeProvider.created = 0
    monkeypatch.setattr(module, "TracerProvider", FakeProvider)
    monkeypatch.setattr(module, "BatchSpanProcessor", lambda exporter: ("processor", exporter))
    monkeypatch.setattr(module, "_SilentOTLPSpanExporter", Mock(return_value="exporter"))
    set_provider = Mock()
    monkeypatch.setattr(module.trace, "set_tracer_provider", set_provider)

    ModunaOTEL("agent-a", "langchain", api_key="key", auto_shutdown=False)
    ModunaOTEL("agent-b", "langchain", api_key="key", auto_shutdown=False)

    assert FakeProvider.created == 1
    assert set_provider.call_count == 1


def test_constructor_accepts_mapping_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dict-style config can be used for one-import/one-instantiation setup."""
    FakeProvider.created = 0
    monkeypatch.setattr(module, "TracerProvider", FakeProvider)
    monkeypatch.setattr(module, "BatchSpanProcessor", lambda exporter: ("processor", exporter))
    monkeypatch.setattr(module, "_SilentOTLPSpanExporter", Mock(return_value="exporter"))
    monkeypatch.setattr(module.trace, "set_tracer_provider", Mock())

    otel = ModunaOTEL({"agentName": "agent", "framework": "langchain", "autoShutdown": False})

    assert otel.agent_name == "agent"
    assert otel.framework == "langchain"


def test_constructor_rejects_vercel_ai_sdk_framework() -> None:
    """The Python SDK does not expose Vercel AI SDK support."""
    with pytest.raises(TypeError, match="only supports the 'langchain' framework"):
        ModunaOTEL("agent", "vercel-ai-sdk", auto_shutdown=False)  # type: ignore[arg-type]


def test_mapping_config_rejects_vercel_ai_sdk_framework() -> None:
    """Mapping config rejects non-Python framework names."""
    with pytest.raises(TypeError, match="only supports the 'langchain' framework"):
        ModunaOTEL({"agentName": "agent", "framework": "vercel-ai-sdk", "autoShutdown": False})


def test_instrument_sets_attributes_and_ends_span(monkeypatch: pytest.MonkeyPatch) -> None:
    """Instrument creates a CLIENT span, applies context, and ends it."""
    span = FakeSpan()
    tracer = FakeTracer(span)
    monkeypatch.setattr(module.trace, "get_tracer", Mock(return_value=tracer))
    otel = ModunaOTEL("agent", "langchain", auto_shutdown=False)

    result = otel.instrument(
        "call-model",
        lambda active_span: active_span.attributes["sdk.integration"],
        {"session_id": "s1"},
    )

    assert result == "langchain"
    assert span.attributes["moduna.framework"] == "langchain"
    assert span.attributes["sdk.integration"] == "langchain"
    assert span.attributes["moduna.session.id"] == "s1"
    assert span.ended is True


def test_instrument_records_exception_and_reraises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Instrument records callback exceptions without swallowing them."""
    span = FakeSpan()
    monkeypatch.setattr(module.trace, "get_tracer", Mock(return_value=FakeTracer(span)))
    otel = ModunaOTEL("agent", "langchain", auto_shutdown=False)

    with pytest.raises(ValueError, match="boom"):
        otel.instrument("call-model", lambda: (_ for _ in ()).throw(ValueError("boom")))

    assert len(span.exceptions) == 1
    assert span.ended is True
