"""LangChain callback handlers that emit Moduna OpenTelemetry spans."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from moduna_otel.services.langchain_attributes import (
    apply_completion_attributes,
    apply_invocation_attributes,
    apply_model_attributes,
    apply_prompt_attributes,
    apply_tool_call_attributes,
    apply_tool_result_attributes,
    apply_trace_context,
    apply_usage_attributes,
    set_attribute_if_supported,
)
from moduna_otel.services.langchain_registration import (
    register_global_moduna_langchain_handler,
)
from moduna_otel.services.langchain_runs import (
    ActiveLangChainRun,
    LangChainRunStart,
    apply_run_attributes,
)
from moduna_otel.services.langchain_utils import (
    count_generations,
    dumps,
    get_record,
    get_string,
    get_value,
    normalize_message,
    serialized_id,
)
from moduna_otel.types import ModunaTraceContext

__all__ = [
    "ModunaAsyncLangChainCallbackHandler",
    "ModunaLangChainCallbackHandler",
    "register_global_moduna_langchain_handler",
]

if TYPE_CHECKING:
    from langchain_core.callbacks import (
        AsyncCallbackHandler as _AsyncBaseCallbackHandler,
    )
    from langchain_core.callbacks import (
        BaseCallbackHandler as _BaseCallbackHandler,
    )
else:
    try:  # LangChain is optional so non-LangChain users keep a small install.
        from langchain_core.callbacks import (
            AsyncCallbackHandler as _AsyncBaseCallbackHandler,
        )
        from langchain_core.callbacks import (
            BaseCallbackHandler as _BaseCallbackHandler,
        )
    except Exception:  # pragma: no cover - exercised only without langchain-core.

        class _BaseCallbackHandler:
            """Fallback base class used when langchain-core is not installed."""

        class _AsyncBaseCallbackHandler:
            """Fallback async base class used when langchain-core is not installed."""


class _ModunaLangChainTracingCore:
    """Shared tracing implementation for sync and async callback handlers."""

    trace_context: ModunaTraceContext
    debug: bool
    logger: logging.Logger
    runs: dict[str, ActiveLangChainRun]

    def _start_model_run(
        self,
        run: LangChainRunStart,
    ) -> None:
        """Create and populate a model span for an active LangChain run."""
        span = self._start_span(run.run_name or "langchain.llm")
        apply_run_attributes(span, run)
        apply_model_attributes(span, run.serialized, run.extra_params)
        apply_prompt_attributes(span, run.input_messages)
        apply_invocation_attributes(span, run.extra_params)
        apply_trace_context(span, run.metadata, self.trace_context)
        self._store_run(run.run_id, span, run.run_type)
        self._debug_log("start", span, run_id=run.run_id)

    def _start_tool_run(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Any | None,
        tags: list[str] | None,
        metadata: dict[str, Any] | None,
        inputs: dict[str, Any] | None,
        extra_params: dict[str, Any],
    ) -> None:
        """Create and populate a tool execution span."""
        name = extra_params.get("name")
        tool_run_name = name if isinstance(name, str) else None
        tool_name = tool_run_name or _serialized_name(serialized) or "tool"
        span = self._start_span(f"execute_tool {tool_name}")
        run = LangChainRunStart(
            serialized=serialized,
            input_messages=[],
            input_count=1 if input_str or inputs else 0,
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            tags=tags,
            metadata=metadata,
            run_name=tool_run_name,
            run_type="tool",
            extra_params=extra_params,
        )
        apply_run_attributes(span, run)
        apply_tool_call_attributes(
            span,
            tool_name,
            inputs if inputs is not None else input_str,
            metadata,
            extra_params,
        )
        apply_trace_context(span, metadata, self.trace_context)
        self._store_run(run.run_id, span, run.run_type)
        self._debug_log("tool_start", span, run_id=run.run_id)

    def _start_chain_run(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any | None,
        tags: list[str] | None,
        metadata: dict[str, Any] | None,
        extra_params: dict[str, Any],
    ) -> None:
        """Create and populate a chain or agent span."""
        run_name = _string_param(extra_params, "name") or _serialized_name(serialized)
        span = self._start_span(run_name or "langchain.chain")
        run = LangChainRunStart(
            serialized=serialized,
            input_messages=[],
            input_count=len(inputs),
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            tags=tags,
            metadata=metadata,
            run_name=run_name,
            run_type="chain",
            extra_params=extra_params,
        )
        apply_run_attributes(span, run)
        span.set_attribute("langchain.serialized.id", serialized_id(serialized))
        set_attribute_if_supported(span, "langchain.input", inputs)
        apply_trace_context(span, metadata, self.trace_context)
        self._store_run(run.run_id, span, run.run_type)
        self._debug_log("chain_start", span, run_id=run.run_id)

    def _start_retriever_run(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: Any,
        parent_run_id: Any | None,
        tags: list[str] | None,
        metadata: dict[str, Any] | None,
        extra_params: dict[str, Any],
    ) -> None:
        """Create and populate a retriever span."""
        run_name = _string_param(extra_params, "name") or _serialized_name(serialized)
        span = self._start_span(run_name or "langchain.retriever")
        run = LangChainRunStart(
            serialized=serialized,
            input_messages=[{"role": "user", "content": query}],
            input_count=1,
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            tags=tags,
            metadata=metadata,
            run_name=run_name,
            run_type="retriever",
            extra_params=extra_params,
        )
        apply_run_attributes(span, run)
        span.set_attribute("langchain.serialized.id", serialized_id(serialized))
        span.set_attribute("gen_ai.retrieval.query", query)
        apply_prompt_attributes(span, run.input_messages)
        apply_trace_context(span, metadata, self.trace_context)
        self._store_run(run.run_id, span, run.run_type)
        self._debug_log("retriever_start", span, run_id=run.run_id)

    def _end_llm_run(self, response: Any, *, run_id: Any) -> None:
        """Record completion and usage attributes, then end an LLM span."""
        run_key = str(run_id)
        run = self.runs.get(run_key)
        if run is None:
            return

        generations = get_value(response, "generations", []) or []
        run.span.set_attribute("langchain.output.generations", len(generations))
        run.span.set_attribute("langchain.output.candidates", count_generations(generations))
        apply_completion_attributes(run.span, response)
        apply_usage_attributes(run.span, response, run.streamed_token_count)
        self._end_run(run_key, Status(StatusCode.OK), "end")

    def _end_tool_run(self, output: Any, *, run_id: Any) -> None:
        """Record tool output attributes, then end a tool span."""
        run_key = str(run_id)
        run = self.runs.get(run_key)
        if run is None:
            return

        apply_tool_result_attributes(run.span, output)
        self._end_run(run_key, Status(StatusCode.OK), "tool_end")

    def _end_chain_run(self, outputs: dict[str, Any], *, run_id: Any) -> None:
        """Record chain outputs, then end a chain span."""
        run_key = str(run_id)
        run = self.runs.get(run_key)
        if run is None:
            return

        set_attribute_if_supported(run.span, "langchain.output", outputs)
        self._end_run(run_key, Status(StatusCode.OK), "chain_end")

    def _end_retriever_run(self, documents: Sequence[Any], *, run_id: Any) -> None:
        """Record retriever documents, then end a retriever span."""
        run_key = str(run_id)
        run = self.runs.get(run_key)
        if run is None:
            return

        run.span.set_attribute("gen_ai.retrieval.documents.count", len(documents))
        set_attribute_if_supported(
            run.span,
            "gen_ai.retrieval.documents",
            [_document_payload(document) for document in documents],
        )
        self._end_run(run_key, Status(StatusCode.OK), "retriever_end")

    def _record_llm_token(self, token: str, *, run_id: Any) -> None:
        """Record a streamed token event on the active run span."""
        run = self.runs.get(str(run_id))
        if run is None:
            return
        run.streamed_token_count += 1
        run.span.add_event("gen_ai.content.completion", {"content": token, "role": "assistant"})
        self._debug_log("token", run.span, run_id=str(run_id))

    def _record_error(self, error: BaseException, *, run_id: Any, debug_event: str) -> None:
        """Record an error and end the active span."""
        run_key = str(run_id)
        run = self.runs.get(run_key)
        if run is None:
            return

        run.span.set_attribute("error.type", type(error).__name__)
        run.span.set_attribute("error.message", str(error))
        run.span.record_exception(error)
        self._end_run(run_key, Status(StatusCode.ERROR, str(error)), debug_event)

    def _record_or_create_event(
        self,
        name: str,
        payload: dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Attach an event to an active run, or emit a short-lived event span."""
        run_key = str(run_id)
        run = self.runs.get(run_key)
        event_payload = _event_payload(payload)
        if run is not None:
            run.span.add_event(name, event_payload)
            self._debug_log(name, run.span, run_id=run_key)
            return

        span = self._start_span(f"langchain.{name}")
        apply_run_attributes(
            span,
            LangChainRunStart(
                serialized={},
                input_messages=[],
                input_count=0,
                run_id=run_key,
                parent_run_id=str(parent_run_id) if parent_run_id else None,
                tags=tags,
                metadata=metadata,
                run_name=name,
                run_type="event",
                extra_params={},
            ),
        )
        apply_trace_context(span, metadata, self.trace_context)
        span.add_event(name, event_payload)
        span.set_status(Status(StatusCode.OK))
        span.end()
        self._debug_log(name, span, run_id=run_key)

    def _start_span(self, name: str) -> Span:
        """Start a CLIENT span using Moduna's LangChain tracer."""
        return trace.get_tracer("moduna-langchain").start_span(name, kind=SpanKind.CLIENT)

    def _store_run(self, run_id: str, span: Span, run_type: str) -> None:
        """Store active run state."""
        self.runs[run_id] = ActiveLangChainRun(
            span=span,
            streamed_token_count=0,
            run_type=run_type,
        )

    def _end_run(self, run_key: str, status: Status, debug_event: str) -> None:
        """Set status, end, and remove an active run."""
        run = self.runs[run_key]
        run.span.set_status(status)
        run.span.end()
        del self.runs[run_key]
        self._debug_log(debug_event, run.span, run_id=run_key)

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


class ModunaLangChainCallbackHandler(_BaseCallbackHandler, _ModunaLangChainTracingCore):
    """LangChain callback handler that creates Moduna spans for sync callbacks."""

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
        self._start_model_run(
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
        self._start_model_run(
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
        self._record_llm_token(token, run_id=run_id)

    def on_llm_end(self, response: Any, *, run_id: Any, **_: Any) -> None:
        """Record completion and usage attributes, then end the active span."""
        self._end_llm_run(response, run_id=run_id)

    def on_llm_error(self, error: BaseException, *, run_id: Any, **_: Any) -> None:
        """Record a LangChain error and end the active span."""
        self._record_error(error, run_id=run_id, debug_event="error")

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Start a CLIENT span for a LangChain chain run."""
        self._start_chain_run(
            serialized,
            inputs,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            extra_params=kwargs,
        )

    def on_chain_end(self, outputs: dict[str, Any], *, run_id: Any, **_: Any) -> None:
        """Record chain outputs, then end the active chain span."""
        self._end_chain_run(outputs, run_id=run_id)

    def on_chain_error(self, error: BaseException, *, run_id: Any, **_: Any) -> None:
        """Record a LangChain chain error and end the active chain span."""
        self._record_error(error, run_id=run_id, debug_event="chain_error")

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Start a CLIENT span for a LangChain tool execution."""
        self._start_tool_run(
            serialized,
            input_str,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            inputs=inputs,
            extra_params=kwargs,
        )

    def on_tool_end(self, output: Any, *, run_id: Any, **_: Any) -> None:
        """Record tool output attributes, then end the active tool span."""
        self._end_tool_run(output, run_id=run_id)

    def on_tool_error(self, error: BaseException, *, run_id: Any, **_: Any) -> None:
        """Record a LangChain tool error and end the active tool span."""
        self._record_error(error, run_id=run_id, debug_event="tool_error")

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Start a CLIENT span for a LangChain retriever run."""
        self._start_retriever_run(
            serialized,
            query,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            extra_params=kwargs,
        )

    def on_retriever_end(self, documents: Sequence[Any], *, run_id: Any, **_: Any) -> None:
        """Record retrieved documents, then end the active retriever span."""
        self._end_retriever_run(documents, run_id=run_id)

    def on_retriever_error(self, error: BaseException, *, run_id: Any, **_: Any) -> None:
        """Record a LangChain retriever error and end the active retriever span."""
        self._record_error(error, run_id=run_id, debug_event="retriever_error")

    # CamelCase aliases mirror the TypeScript service and help direct usage.
    handleChatModelStart: ClassVar[Callable[..., None]] = on_chat_model_start  # noqa: N815
    handleLLMStart: ClassVar[Callable[..., None]] = on_llm_start  # noqa: N815
    handleLLMNewToken: ClassVar[Callable[..., None]] = on_llm_new_token  # noqa: N815
    handleLLMEnd: ClassVar[Callable[..., None]] = on_llm_end  # noqa: N815
    handleLLMError: ClassVar[Callable[..., None]] = on_llm_error  # noqa: N815
    handleChainStart: ClassVar[Callable[..., None]] = on_chain_start  # noqa: N815
    handleChainEnd: ClassVar[Callable[..., None]] = on_chain_end  # noqa: N815
    handleChainError: ClassVar[Callable[..., None]] = on_chain_error  # noqa: N815
    handleToolStart: ClassVar[Callable[..., None]] = on_tool_start  # noqa: N815
    handleToolEnd: ClassVar[Callable[..., None]] = on_tool_end  # noqa: N815
    handleToolError: ClassVar[Callable[..., None]] = on_tool_error  # noqa: N815
    handleRetrieverStart: ClassVar[Callable[..., None]] = on_retriever_start  # noqa: N815
    handleRetrieverEnd: ClassVar[Callable[..., None]] = on_retriever_end  # noqa: N815
    handleRetrieverError: ClassVar[Callable[..., None]] = on_retriever_error  # noqa: N815


class ModunaAsyncLangChainCallbackHandler(
    _AsyncBaseCallbackHandler,
    _ModunaLangChainTracingCore,
):
    """LangChain callback handler that creates Moduna spans for async callbacks."""

    name: ClassVar[str] = "moduna_otel_async_langchain_callback_handler"

    def __init__(
        self,
        trace_context: ModunaTraceContext | None = None,
        debug: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        """Create an async handler with optional default conversation/session context."""
        super().__init__()
        self.trace_context: ModunaTraceContext = trace_context or {}
        self.debug: bool = debug
        self.logger: logging.Logger = logger or logging.getLogger("moduna_otel.langchain")
        self.runs: dict[str, ActiveLangChainRun] = {}

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Start a CLIENT span for an async LangChain text LLM run."""
        self._start_model_run(
            LangChainRunStart(
                serialized=serialized,
                input_messages=[{"role": "user", "content": prompt} for prompt in prompts],
                input_count=len(prompts),
                run_id=str(run_id),
                parent_run_id=str(parent_run_id) if parent_run_id else None,
                tags=tags,
                metadata=metadata,
                run_name=_string_param(kwargs, "name"),
                run_type="llm",
                extra_params=kwargs,
            )
        )

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Start a CLIENT span for an async LangChain chat model run."""
        flat_messages = [normalize_message(message) for batch in messages for message in batch]
        self._start_model_run(
            LangChainRunStart(
                serialized=serialized,
                input_messages=flat_messages,
                input_count=len(flat_messages),
                run_id=str(run_id),
                parent_run_id=str(parent_run_id) if parent_run_id else None,
                tags=tags,
                metadata=metadata,
                run_name=_string_param(kwargs, "name"),
                run_type="chat_model",
                extra_params=kwargs,
            )
        )

    async def on_llm_new_token(self, token: str, *, run_id: Any, **_: Any) -> None:
        """Record a streamed token event on the active async run span."""
        self._record_llm_token(token, run_id=run_id)

    async def on_llm_end(self, response: Any, *, run_id: Any, **_: Any) -> None:
        """Record async completion and usage attributes, then end the active span."""
        self._end_llm_run(response, run_id=run_id)

    async def on_llm_error(self, error: BaseException, *, run_id: Any, **_: Any) -> None:
        """Record an async LangChain LLM error and end the active span."""
        self._record_error(error, run_id=run_id, debug_event="error")

    async def on_stream_event(
        self,
        event: Any,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        **_: Any,
    ) -> None:
        """Record an async stream protocol event."""
        self._record_or_create_event(
            "langchain.stream",
            {"event": event},
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
        )

    async def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Start a CLIENT span for an async LangChain chain run."""
        self._start_chain_run(
            serialized,
            inputs,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            extra_params=kwargs,
        )

    async def on_chain_end(self, outputs: dict[str, Any], *, run_id: Any, **_: Any) -> None:
        """Record async chain outputs, then end the active chain span."""
        self._end_chain_run(outputs, run_id=run_id)

    async def on_chain_error(self, error: BaseException, *, run_id: Any, **_: Any) -> None:
        """Record an async chain error and end the active chain span."""
        self._record_error(error, run_id=run_id, debug_event="chain_error")

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Start a CLIENT span for an async LangChain tool execution."""
        self._start_tool_run(
            serialized,
            input_str,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            inputs=inputs,
            extra_params=kwargs,
        )

    async def on_tool_end(self, output: Any, *, run_id: Any, **_: Any) -> None:
        """Record async tool output attributes, then end the active tool span."""
        self._end_tool_run(output, run_id=run_id)

    async def on_tool_error(self, error: BaseException, *, run_id: Any, **_: Any) -> None:
        """Record an async tool error and end the active tool span."""
        self._record_error(error, run_id=run_id, debug_event="tool_error")

    async def on_text(
        self,
        text: str,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        **_: Any,
    ) -> None:
        """Record arbitrary async LangChain text."""
        self._record_or_create_event(
            "langchain.text",
            {"text": text},
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
        )

    async def on_retry(
        self,
        retry_state: Any,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        **_: Any,
    ) -> None:
        """Record an async LangChain retry event."""
        self._record_or_create_event(
            "langchain.retry",
            {"retry_state": retry_state},
            run_id=run_id,
            parent_run_id=parent_run_id,
        )

    async def on_agent_action(
        self,
        action: Any,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        **_: Any,
    ) -> None:
        """Record an async agent action event."""
        self._record_or_create_event(
            "langchain.agent.action",
            {"action": _agent_payload(action)},
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
        )

    async def on_agent_finish(
        self,
        finish: Any,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        **_: Any,
    ) -> None:
        """Record an async agent finish event."""
        self._record_or_create_event(
            "langchain.agent.finish",
            {"finish": _agent_payload(finish)},
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
        )

    async def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Start a CLIENT span for an async LangChain retriever run."""
        self._start_retriever_run(
            serialized,
            query,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            extra_params=kwargs,
        )

    async def on_retriever_end(
        self,
        documents: Sequence[Any],
        *,
        run_id: Any,
        **_: Any,
    ) -> None:
        """Record async retrieved documents, then end the active retriever span."""
        self._end_retriever_run(documents, run_id=run_id)

    async def on_retriever_error(self, error: BaseException, *, run_id: Any, **_: Any) -> None:
        """Record an async retriever error and end the active retriever span."""
        self._record_error(error, run_id=run_id, debug_event="retriever_error")

    async def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: Any,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        """Record an async custom LangChain event."""
        self._record_or_create_event(
            f"langchain.custom.{name}",
            {"name": name, "data": data},
            run_id=run_id,
            tags=tags,
            metadata=metadata,
        )


def _serialized_name(serialized: dict[str, Any]) -> str | None:
    """Find a LangChain runnable name from serialized callback metadata."""
    value = serialized.get("name")
    if isinstance(value, str):
        return value
    serialized_value = serialized.get("id")
    if isinstance(serialized_value, list) and serialized_value:
        return str(serialized_value[-1])
    if serialized_value:
        return str(serialized_value)
    return None


def _string_param(value: dict[str, Any], key: str) -> str | None:
    """Return a string parameter from a callback kwargs dict."""
    result = value.get(key)
    return result if isinstance(result, str) else None


def _document_payload(document: Any) -> dict[str, Any]:
    """Return a compact serializable retriever document payload."""
    metadata = get_record(document, "metadata") or {}
    page_content = get_string(document, "page_content") or get_string(document, "content")
    payload: dict[str, Any] = {}
    if page_content is not None:
        payload["page_content"] = page_content
    if metadata:
        payload["metadata"] = metadata
    if not payload:
        payload["value"] = str(document)
    return payload


def _agent_payload(value: Any) -> Any:
    """Return a useful serializable payload for agent callbacks."""
    if isinstance(value, dict):
        return value
    payload: dict[str, Any] = {}
    for key in ("tool", "tool_input", "log", "return_values"):
        item = get_value(value, key)
        if item is not None:
            payload[key] = item
    return payload or str(value)


def _event_payload(payload: dict[str, Any]) -> dict[str, str | bool | int | float]:
    """Normalize event payload values to OpenTelemetry-supported event attributes."""
    normalized: dict[str, str | bool | int | float] = {}
    for key, value in payload.items():
        if isinstance(value, str | bool | int | float):
            normalized[key] = value
        else:
            normalized[key] = dumps(value)
    return normalized
