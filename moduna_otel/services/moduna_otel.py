"""One-line OpenTelemetry setup for Moduna AI traces."""

from __future__ import annotations

import atexit
import inspect
import logging
import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar, overload

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExportResult
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from moduna_otel.services.moduna_langchain_callback_handler import (
    ModunaLangChainCallbackHandler,
    register_global_moduna_langchain_handler,
)
from moduna_otel.types import Framework, ModunaOTELConfig, ModunaTraceContext

DEFAULT_ENDPOINT = "https://volex-otel-git-506013021984.us-central1.run.app/v1/traces"
T = TypeVar("T")


@dataclass(slots=True)
class _SharedState:
    """Mutable singleton state shared by all ModunaOTEL instances."""

    provider: TracerProvider | None = None
    started: bool = False
    shutdown_started: bool = False
    lifecycle_hooks_registered: bool = False
    warned: bool = False


class _SilentOTLPSpanExporter:
    """OTLP exporter wrapper that converts telemetry failures into warnings."""

    def __init__(
        self,
        endpoint: str,
        headers: Mapping[str, str],
        on_failure: Callable[[BaseException], None],
    ) -> None:
        """Create the wrapped OTLP HTTP exporter."""
        self._exporter = OTLPSpanExporter(endpoint=endpoint, headers=dict(headers))
        self._on_failure = on_failure

    def export(self, spans: Any) -> SpanExportResult:
        """Export spans while preventing exporter exceptions from escaping."""
        try:
            result = self._exporter.export(spans)
        except BaseException as exc:
            self._on_failure(exc)
            return SpanExportResult.FAILURE

        if result is not SpanExportResult.SUCCESS:
            self._on_failure(RuntimeError(f"Moduna OTEL exporter failed: {result!r}"))
        return result

    def shutdown(self) -> None:
        """Shut down the wrapped exporter without raising telemetry errors."""
        try:
            self._exporter.shutdown()
        except BaseException as exc:
            self._on_failure(exc)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Flush the wrapped exporter if it supports flushing."""
        flush = getattr(self._exporter, "force_flush", None)
        if flush is None:
            return True
        try:
            return bool(flush(timeout_millis))
        except BaseException as exc:
            self._on_failure(exc)
            return False


class ModunaOTEL:
    """Ergonomic Moduna OpenTelemetry integration for AI applications."""

    _shared = _SharedState()
    _logger = logging.getLogger("moduna_otel")

    def __init__(
        self,
        agent_name: str | ModunaOTELConfig | Mapping[str, Any],
        framework: Framework | None = None,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
        auto_shutdown: bool = True,
    ) -> None:
        """Create a Moduna telemetry wrapper and start tracing immediately."""
        config = self._normalize_config(
            agent_name=agent_name,
            framework=framework,
            api_key=api_key,
            headers=headers,
            auto_shutdown=auto_shutdown,
        )
        self.agent_name = config.agent_name
        self.framework = config.framework

        if ModunaOTEL._shared.provider is None:
            ModunaOTEL._shared.provider = self._create_provider(config)

        self._register_lifecycle_hooks(config)
        self.start()

    @classmethod
    def start_from_config(cls, config: ModunaOTELConfig) -> ModunaOTEL:
        """Create and start a ModunaOTEL instance from a config object."""
        return cls(config)

    def start(self) -> None:
        """Start singleton OpenTelemetry provider setup if it has not run yet."""
        if ModunaOTEL._shared.started:
            return

        try:
            provider = ModunaOTEL._shared.provider
            if provider is not None:
                trace.set_tracer_provider(provider)
            ModunaOTEL._shared.started = True
        except BaseException as exc:
            self._warn_once(exc)

    def shutdown(self) -> None:
        """Flush and shut down the singleton OpenTelemetry provider."""
        shared = ModunaOTEL._shared
        if shared.shutdown_started or shared.provider is None:
            return

        shared.shutdown_started = True
        try:
            shared.provider.shutdown()
        except BaseException as exc:
            self._warn_once(exc)
        finally:
            shared.started = False
            shared.provider = None
            shared.shutdown_started = False

    @overload
    def instrument(
        self,
        span_name: str,
        callback: Callable[[Span], T] | Callable[[], T],
        context: ModunaTraceContext | None = None,
    ) -> T: ...

    @overload
    def instrument(
        self,
        span_name: str,
        callback: Callable[[Span], Awaitable[T]] | Callable[[], Awaitable[T]],
        context: ModunaTraceContext | None = None,
    ) -> Awaitable[T]: ...

    def instrument(
        self,
        span_name: str,
        callback: Callable[..., T] | Callable[..., Awaitable[T]],
        context: ModunaTraceContext | None = None,
    ) -> T | Awaitable[T]:
        """Run a callback inside a CLIENT span and record failures on the span."""
        tracer = trace.get_tracer("moduna-gen-ai")
        span = tracer.start_span(span_name, kind=SpanKind.CLIENT)
        self._apply_common_span_attributes(span, context or {})

        try:
            result = self._call_with_optional_span(callback, span)
        except BaseException as exc:
            self._record_span_exception(span, exc)
            span.end()
            raise

        if inspect.isawaitable(result):
            return self._finish_async_result(result, span)

        span.end()
        return result

    def vercel_telemetry(
        self,
        context: ModunaTraceContext | None = None,
    ) -> dict[str, Any]:
        """Return Vercel AI SDK compatible metadata-style telemetry settings."""
        return {
            "isEnabled": True,
            "metadata": self._trace_metadata(context or {}),
        }

    def langchain_handler(
        self,
        context: ModunaTraceContext | None = None,
        **config: Any,
    ) -> ModunaLangChainCallbackHandler:
        """Create a LangChain callback handler with optional default context."""
        return ModunaLangChainCallbackHandler(trace_context=context or {}, **config)

    def register_global_langchain_handler(
        self,
        context: ModunaTraceContext | None = None,
        **config: Any,
    ) -> ModunaLangChainCallbackHandler:
        """Best-effort global LangChain callback registration helper."""
        handler = self.langchain_handler(context, **config)
        register_global_moduna_langchain_handler(handler, self._warn_once)
        return handler

    @classmethod
    def _reset_for_tests(cls) -> None:
        """Reset singleton state for isolated unit tests."""
        cls._shared = _SharedState()

    @classmethod
    def _warn_once(cls, error: BaseException) -> None:
        """Log a single warning for telemetry failures."""
        if cls._shared.warned:
            return
        cls._shared.warned = True
        cls._logger.warning("Moduna OTEL failed to send telemetry.", exc_info=error)

    def _create_provider(self, config: ModunaOTELConfig) -> TracerProvider:
        """Create the singleton OpenTelemetry tracer provider."""
        resource = Resource.create(
            {
                "service.name": config.agent_name,
                "moduna.framework": config.framework,
                "sdk.integration": config.framework,
            }
        )
        provider = TracerProvider(resource=resource)
        exporter = _SilentOTLPSpanExporter(
            endpoint=DEFAULT_ENDPOINT,
            headers=self._export_headers(config),
            on_failure=self._warn_once,
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        return provider

    def _export_headers(self, config: ModunaOTELConfig) -> dict[str, str]:
        """Build OTLP HTTP headers, including Authorization when configured."""
        export_headers: dict[str, str] = {}
        if config.api_key:
            export_headers["Authorization"] = f"Bearer {config.api_key}"
        if config.headers:
            export_headers.update(config.headers)
        return export_headers

    def _normalize_config(
        self,
        agent_name: str | ModunaOTELConfig | Mapping[str, Any],
        framework: Framework | None,
        api_key: str | None,
        headers: dict[str, str] | None,
        auto_shutdown: bool,
    ) -> ModunaOTELConfig:
        """Normalize constructor input into a typed config object."""
        if isinstance(agent_name, ModunaOTELConfig):
            return ModunaOTELConfig(
                agent_name=agent_name.agent_name,
                framework=agent_name.framework,
                api_key=agent_name.api_key or os.getenv("MODUNA_API_KEY"),
                headers=agent_name.headers,
                auto_shutdown=agent_name.auto_shutdown,
            )

        if isinstance(agent_name, Mapping):
            mapped_framework = agent_name.get("framework", framework)
            if mapped_framework not in ("langchain", "vercel-ai-sdk"):
                raise TypeError("ModunaOTEL config requires a supported framework.")
            mapped_agent_name = agent_name.get("agent_name") or agent_name.get("agentName")
            if not isinstance(mapped_agent_name, str):
                raise TypeError("ModunaOTEL config requires agent_name.")
            mapped_headers = agent_name.get("headers", headers)
            if mapped_headers is not None and not isinstance(mapped_headers, dict):
                raise TypeError("ModunaOTEL headers must be a dict[str, str].")
            return ModunaOTELConfig(
                agent_name=mapped_agent_name,
                framework=mapped_framework,
                api_key=self._optional_string(
                    agent_name.get("api_key") or agent_name.get("apiKey") or api_key
                )
                or os.getenv("MODUNA_API_KEY"),
                headers=mapped_headers,
                auto_shutdown=bool(
                    agent_name.get("auto_shutdown", agent_name.get("autoShutdown", auto_shutdown))
                ),
            )

        if framework is None:
            raise TypeError("ModunaOTEL requires a framework.")

        return ModunaOTELConfig(
            agent_name=agent_name,
            framework=framework,
            api_key=api_key or os.getenv("MODUNA_API_KEY"),
            headers=headers,
            auto_shutdown=auto_shutdown,
        )

    def _optional_string(self, value: Any) -> str | None:
        """Return a value only when it is a string."""
        return value if isinstance(value, str) else None

    def _register_lifecycle_hooks(self, config: ModunaOTELConfig) -> None:
        """Register process-exit shutdown once when auto shutdown is enabled."""
        shared = ModunaOTEL._shared
        if not config.auto_shutdown or shared.lifecycle_hooks_registered:
            return
        atexit.register(self.shutdown)
        shared.lifecycle_hooks_registered = True

    def _trace_metadata(self, context: ModunaTraceContext) -> dict[str, str]:
        """Convert Moduna context aliases into OpenTelemetry metadata keys."""
        metadata: dict[str, str] = {}
        conversation_id = context.get("conversation_id") or context.get("conversationId")
        session_id = context.get("session_id") or context.get("sessionId")
        if conversation_id:
            metadata["moduna.conversation.id"] = conversation_id
        if session_id:
            metadata["moduna.session.id"] = session_id
        return metadata

    def _apply_common_span_attributes(
        self,
        span: Span,
        context: ModunaTraceContext,
    ) -> None:
        """Attach framework and per-call context attributes to a span."""
        span.set_attribute("moduna.framework", self.framework)
        span.set_attribute("sdk.integration", self.framework)
        for key, value in self._trace_metadata(context).items():
            span.set_attribute(key, value)

    def _call_with_optional_span(self, callback: Callable[..., Any], span: Span) -> Any:
        """Call callbacks that accept either no arguments or a span argument."""
        try:
            signature = inspect.signature(callback)
        except (TypeError, ValueError):
            return callback(span)

        accepts_positional = any(
            parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
            for parameter in signature.parameters.values()
        )
        accepts_varargs = any(
            parameter.kind is parameter.VAR_POSITIONAL
            for parameter in signature.parameters.values()
        )
        return callback(span) if accepts_positional or accepts_varargs else callback()

    async def _finish_async_result(self, result: Awaitable[T], span: Span) -> T:
        """Await an async callback result and end the span afterwards."""
        try:
            return await result
        except BaseException as exc:
            self._record_span_exception(span, exc)
            raise
        finally:
            span.end()

    def _record_span_exception(self, span: Span, exc: BaseException) -> None:
        """Record a callback exception and mark the span as failed."""
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, str(exc)))
