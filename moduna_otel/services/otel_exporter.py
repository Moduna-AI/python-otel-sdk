"""OpenTelemetry exporter helpers for Moduna."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


class SilentOTLPSpanExporter(SpanExporter):
    """OTLP exporter wrapper that converts telemetry failures into warnings."""

    def __init__(
        self,
        endpoint: str,
        headers: dict[str, str],
        on_failure: Callable[[BaseException], None],
    ) -> None:
        """Create the wrapped OTLP HTTP exporter."""
        self._exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)
        self._on_failure = on_failure

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
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
        try:
            return bool(self._exporter.force_flush(timeout_millis))
        except BaseException as exc:
            self._on_failure(exc)
            return False
