"""One-line OpenTelemetry setup for Moduna AI traces."""

from __future__ import annotations

import atexit
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, ClassVar, TypeVar, cast, overload

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, SpanKind

from moduna_otel.services.moduna_langchain_callback_handler import (
    ModunaLangChainCallbackHandler,
    register_global_moduna_langchain_handler,
)
from moduna_otel.services.otel_config import (
    DEFAULT_ENDPOINT,
    SharedOTELState,
    export_headers,
    normalize_config,
)
from moduna_otel.services.otel_exporter import SilentOTLPSpanExporter
from moduna_otel.services.otel_instrumentation import (
    apply_common_span_attributes,
    call_with_optional_span,
    finish_async_result,
    record_span_exception,
)
from moduna_otel.types import Framework, ModunaOTELConfig, ModunaTraceContext

T = TypeVar("T")


class ModunaOTEL:
    """Ergonomic Moduna OpenTelemetry integration for AI applications."""

    _shared: ClassVar[SharedOTELState] = SharedOTELState()
    _logger: ClassVar[logging.Logger] = logging.getLogger("moduna_otel")

    def __init__(
        self,
        agent_name: str | ModunaOTELConfig | Mapping[str, Any],
        framework: Framework | None = None,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
        auto_shutdown: bool = True,
    ) -> None:
        """Create a Moduna telemetry wrapper and start tracing immediately."""
        config = normalize_config(agent_name, framework, api_key, headers, auto_shutdown)
        self.agent_name: str = config.agent_name
        self.framework: Framework = config.framework

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
        callback: Callable[[Span], Awaitable[T]],
        context: ModunaTraceContext | None = None,
    ) -> Awaitable[T]: ...

    @overload
    def instrument(
        self,
        span_name: str,
        callback: Callable[[], Awaitable[T]],
        context: ModunaTraceContext | None = None,
    ) -> Awaitable[T]: ...

    @overload
    def instrument(
        self,
        span_name: str,
        callback: Callable[[Span], T],
        context: ModunaTraceContext | None = None,
    ) -> T: ...

    @overload
    def instrument(
        self,
        span_name: str,
        callback: Callable[[], T],
        context: ModunaTraceContext | None = None,
    ) -> T: ...

    def instrument(
        self,
        span_name: str,
        callback: Callable[..., T] | Callable[..., Awaitable[T]],
        context: ModunaTraceContext | None = None,
    ) -> T | Awaitable[T]:
        """Run a callback inside a CLIENT span and record failures on the span."""
        span = trace.get_tracer("moduna-gen-ai").start_span(span_name, kind=SpanKind.CLIENT)
        apply_common_span_attributes(span, self.framework, context or {})

        try:
            result = cast(T | Awaitable[T], call_with_optional_span(callback, span))
        except BaseException as exc:
            record_span_exception(span, exc)
            span.end()
            raise

        if inspect.isawaitable(result):
            return finish_async_result(result, span)

        span.end()
        return result

    def langchain_handler(
        self,
        context: ModunaTraceContext | None = None,
        *,
        debug: bool = False,
        logger: logging.Logger | None = None,
    ) -> ModunaLangChainCallbackHandler:
        """Create a LangChain callback handler with optional default context."""
        return ModunaLangChainCallbackHandler(
            trace_context=context or {},
            debug=debug,
            logger=logger,
        )

    def register_global_langchain_handler(
        self,
        context: ModunaTraceContext | None = None,
        *,
        debug: bool = False,
        logger: logging.Logger | None = None,
    ) -> ModunaLangChainCallbackHandler:
        """Best-effort global LangChain callback registration helper."""
        handler = self.langchain_handler(context, debug=debug, logger=logger)
        register_global_moduna_langchain_handler(handler, self._warn_once)
        return handler

    @classmethod
    def _reset_for_tests(cls) -> None:
        """Reset singleton state for isolated unit tests."""
        cls._shared = SharedOTELState()

    @classmethod
    def _warn_once(cls, error: BaseException) -> None:
        """Log a single warning for telemetry failures."""
        if cls._shared.warned:
            return
        cls._shared.warned = True
        cls._logger.warning("Moduna OTEL failed to send telemetry.", exc_info=error)

    def _create_provider(self, config: ModunaOTELConfig) -> TracerProvider:
        """Create the singleton OpenTelemetry tracer provider."""
        provider = TracerProvider(
            resource=Resource.create(
                {
                    "service.name": config.agent_name,
                    "moduna.framework": config.framework,
                    "sdk.integration": config.framework,
                }
            )
        )
        provider.add_span_processor(
            BatchSpanProcessor(
                SilentOTLPSpanExporter(
                    endpoint=DEFAULT_ENDPOINT,
                    headers=export_headers(config),
                    on_failure=self._warn_once,
                )
            )
        )
        return provider

    def _register_lifecycle_hooks(self, config: ModunaOTELConfig) -> None:
        """Register process-exit shutdown once when auto shutdown is enabled."""
        shared = ModunaOTEL._shared
        if not config.auto_shutdown or shared.lifecycle_hooks_registered:
            return
        atexit.register(self.shutdown)
        shared.lifecycle_hooks_registered = True
