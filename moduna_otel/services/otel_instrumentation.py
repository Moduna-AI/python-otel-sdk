"""Manual instrumentation helpers for Moduna OpenTelemetry spans."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from opentelemetry.trace import Span, Status, StatusCode

from moduna_otel.services.otel_config import trace_metadata
from moduna_otel.types import Framework, ModunaTraceContext

T = TypeVar("T")


def apply_common_span_attributes(
    span: Span,
    framework: Framework,
    context: ModunaTraceContext,
) -> None:
    """Attach framework and per-call context attributes to a span."""
    span.set_attribute("moduna.framework", framework)
    span.set_attribute("sdk.integration", framework)
    for key, value in trace_metadata(context).items():
        span.set_attribute(key, value)


def call_with_optional_span(callback: Callable[..., Any], span: Span) -> Any:
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
        parameter.kind is parameter.VAR_POSITIONAL for parameter in signature.parameters.values()
    )
    return callback(span) if accepts_positional or accepts_varargs else callback()


async def finish_async_result(result: Awaitable[T], span: Span) -> T:
    """Await an async callback result and end the span afterwards."""
    try:
        return await result
    except BaseException as exc:
        record_span_exception(span, exc)
        raise
    finally:
        span.end()


def record_span_exception(span: Span, exc: BaseException) -> None:
    """Record a callback exception and mark the span as failed."""
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc)))
