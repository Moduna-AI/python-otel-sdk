"""LangChain callback handler that emits Moduna OpenTelemetry spans."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from moduna_otel.services.langchain_attributes import (
    apply_completion_attributes,
    apply_invocation_attributes,
    apply_model_attributes,
    apply_prompt_attributes,
    apply_trace_context,
    apply_usage_attributes,
)
from moduna_otel.services.langchain_registration import (
    register_global_moduna_langchain_handler,
)
from moduna_otel.services.langchain_runs import (
    ActiveLangChainRun,
    LangChainRunStart,
    apply_run_attributes,
)
from moduna_otel.services.langchain_utils import count_generations, get_value, normalize_message
from moduna_otel.types import ModunaTraceContext

__all__ = ["ModunaLangChainCallbackHandler", "register_global_moduna_langchain_handler"]

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler as _BaseCallbackHandler
else:
    try:  # LangChain is optional so non-LangChain users keep a small install.
        from langchain_core.callbacks import BaseCallbackHandler as _BaseCallbackHandler
    except Exception:  # pragma: no cover - exercised only without langchain-core.

        class _BaseCallbackHandler:
            """Fallback base class used when langchain-core is not installed."""


class ModunaLangChainCallbackHandler(_BaseCallbackHandler):
    """LangChain callback handler that creates Moduna spans for model runs."""

    name: ClassVar[str] = "moduna_otel_langchain_callback_handler"

    def __init__(
        self,
        trace_context: ModunaTraceContext | None = None,
        debug: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        """Create a handler with optional default conversation/session context."""
        super().__init__()
        self.trace_context: ModunaTraceContext = trace_context or {}
        self.debug: bool = debug
        self.logger: logging.Logger = logger or logging.getLogger("moduna_otel.langchain")
        self.runs: dict[str, ActiveLangChainRun] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Start a CLIENT span for a LangChain chat model run."""
        flat_messages = [normalize_message(message) for batch in messages for message in batch]
        self._start_run(
            LangChainRunStart(
                serialized=serialized,
                input_messages=flat_messages,
                input_count=len(flat_messages),
                run_id=str(run_id),
                parent_run_id=str(parent_run_id) if parent_run_id else None,
                tags=tags,
                metadata=metadata,
                run_name=name,
                run_type="chat_model",
                extra_params=kwargs,
            )
        )

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Start a CLIENT span for a LangChain text LLM run."""
        self._start_run(
            LangChainRunStart(
                serialized=serialized,
                input_messages=[{"role": "user", "content": prompt} for prompt in prompts],
                input_count=len(prompts),
                run_id=str(run_id),
                parent_run_id=str(parent_run_id) if parent_run_id else None,
                tags=tags,
                metadata=metadata,
                run_name=name,
                run_type="llm",
                extra_params=kwargs,
            )
        )

    def on_llm_new_token(self, token: str, *, run_id: Any, **_: Any) -> None:
        """Record a streamed token event on the active run span."""
        run = self.runs.get(str(run_id))
        if run is None:
            return
        run.streamed_token_count += 1
        run.span.add_event("gen_ai.content.completion", {"content": token, "role": "assistant"})
        self._debug_log("token", run.span, run_id=str(run_id))

    def on_llm_end(self, response: Any, *, run_id: Any, **_: Any) -> None:
        """Record completion and usage attributes, then end the active span."""
        run_key = str(run_id)
        run = self.runs.get(run_key)
        if run is None:
            return

        generations = get_value(response, "generations", []) or []
        run.span.set_attribute("langchain.output.generations", len(generations))
        run.span.set_attribute("langchain.output.candidates", count_generations(generations))
        apply_completion_attributes(run.span, response)
        apply_usage_attributes(run.span, response, run.streamed_token_count)
        run.span.set_status(Status(StatusCode.OK))
        run.span.end()
        del self.runs[run_key]
        self._debug_log("end", run.span, run_id=run_key)

    def on_llm_error(self, error: BaseException, *, run_id: Any, **_: Any) -> None:
        """Record a LangChain error and end the active span."""
        run_key = str(run_id)
        run = self.runs.get(run_key)
        if run is None:
            return

        run.span.record_exception(error)
        run.span.set_status(Status(StatusCode.ERROR, str(error)))
        run.span.end()
        del self.runs[run_key]
        self._debug_log("error", run.span, run_id=run_key)

    # CamelCase aliases mirror the TypeScript service and help direct usage.
    handleChatModelStart: ClassVar[Callable[..., None]] = on_chat_model_start  # noqa: N815
    handleLLMStart: ClassVar[Callable[..., None]] = on_llm_start  # noqa: N815
    handleLLMNewToken: ClassVar[Callable[..., None]] = on_llm_new_token  # noqa: N815
    handleLLMEnd: ClassVar[Callable[..., None]] = on_llm_end  # noqa: N815
    handleLLMError: ClassVar[Callable[..., None]] = on_llm_error  # noqa: N815

    def _start_run(self, run: LangChainRunStart) -> None:
        """Create and populate a span for an active LangChain run."""
        span = trace.get_tracer("moduna-langchain").start_span(
            run.run_name or "langchain.llm",
            kind=SpanKind.CLIENT,
        )
        apply_run_attributes(span, run)
        apply_model_attributes(span, run.serialized, run.extra_params)
        apply_prompt_attributes(span, run.input_messages)
        apply_invocation_attributes(span, run.extra_params)
        apply_trace_context(span, run.metadata, self.trace_context)
        self.runs[run.run_id] = ActiveLangChainRun(
            span=span,
            streamed_token_count=0,
            run_type=run.run_type,
        )
        self._debug_log("start", span, run_id=run.run_id)

    def _debug_log(self, event: str, span: Span, **payload: Any) -> None:
        """Emit optional debug lifecycle logs for LangChain tracing."""
        if not self.debug:
            return
        span_context = span.get_span_context()
        self.logger.debug(
            "[moduna:langchain] %s",
            {
                "event": event,
                "trace_id": span_context.trace_id,
                "span_id": span_context.span_id,
                **payload,
            },
        )
