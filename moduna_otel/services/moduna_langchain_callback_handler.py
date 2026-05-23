"""LangChain callback handler that emits Moduna OpenTelemetry spans."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from moduna_otel.types import ModunaTraceContext

try:  # LangChain is optional so non-LangChain users keep a small install.
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:  # pragma: no cover - exercised only without langchain-core.

    class BaseCallbackHandler:  # type: ignore[no-redef]
        """Fallback base class used when langchain-core is not installed."""


@dataclass(slots=True)
class _ActiveLangChainRun:
    """OpenTelemetry state for an active LangChain LLM or chat model run."""

    span: Span
    streamed_token_count: int
    run_type: str


class ModunaLangChainCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that creates Moduna spans for model runs."""

    name = "moduna_otel_langchain_callback_handler"

    def __init__(
        self,
        trace_context: ModunaTraceContext | None = None,
        debug: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        """Create a handler with optional default conversation/session context."""
        super().__init__()
        self.trace_context = trace_context or {}
        self.debug = debug
        self.logger = logger or logging.getLogger("moduna_otel.langchain")
        self.runs: dict[str, _ActiveLangChainRun] = {}

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
        flat_messages = [
            self._normalize_message(message) for batch in messages for message in batch
        ]
        self._start_run(
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

    def on_llm_new_token(self, token: str, *, run_id: Any, **_: Any) -> None:
        """Record a streamed token event on the active run span."""
        run = self.runs.get(str(run_id))
        if run is None:
            return
        run.streamed_token_count += 1
        run.span.add_event(
            "gen_ai.content.completion",
            {"content": token, "role": "assistant"},
        )
        self._debug_log("token", run.span, run_id=str(run_id))

    def on_llm_end(self, response: Any, *, run_id: Any, **_: Any) -> None:
        """Record completion and usage attributes, then end the active span."""
        run_key = str(run_id)
        run = self.runs.get(run_key)
        if run is None:
            return

        generations = self._get_value(response, "generations", []) or []
        run.span.set_attribute("langchain.output.generations", len(generations))
        run.span.set_attribute("langchain.output.candidates", self._count_generations(generations))
        self._apply_completion_attributes(run.span, response)
        self._apply_usage_attributes(run.span, response, run.streamed_token_count)
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
    handleChatModelStart = on_chat_model_start  # noqa: N815
    handleLLMStart = on_llm_start  # noqa: N815
    handleLLMNewToken = on_llm_new_token  # noqa: N815
    handleLLMEnd = on_llm_end  # noqa: N815
    handleLLMError = on_llm_error  # noqa: N815

    def _start_run(
        self,
        *,
        serialized: dict[str, Any],
        input_messages: list[dict[str, str]],
        input_count: int,
        run_id: str,
        parent_run_id: str | None,
        tags: list[str] | None,
        metadata: dict[str, Any] | None,
        run_name: str | None,
        run_type: str,
        extra_params: dict[str, Any],
    ) -> None:
        """Create and populate a span for an active LangChain run."""
        span = trace.get_tracer("moduna-langchain").start_span(
            run_name or "langchain.llm",
            kind=SpanKind.CLIENT,
        )
        span.set_attribute("moduna.framework", "langchain")
        span.set_attribute("sdk.integration", "langchain")
        span.set_attribute("langchain.run.id", run_id)
        span.set_attribute("langchain.run.type", run_type)
        span.set_attribute("langchain.input.count", input_count)
        span.set_attribute("langsmith.span.kind", "llm")
        span.set_attribute(
            "gen_ai.operation.name", "chat" if run_type == "chat_model" else "completion"
        )
        span.set_attribute("llm.request.type", "chat" if run_type == "chat_model" else "completion")

        if parent_run_id:
            span.set_attribute("langchain.parent_run.id", parent_run_id)
        if run_name:
            span.set_attribute("langsmith.trace.name", run_name)
        if tags:
            span.set_attribute("langchain.tags", ",".join(tags))
            span.set_attribute("langsmith.span.tags", ",".join(tags))

        self._apply_model_attributes(span, serialized, extra_params)
        self._apply_prompt_attributes(span, input_messages)
        self._apply_invocation_attributes(span, extra_params)
        self._apply_trace_context(span, metadata)
        self.runs[run_id] = _ActiveLangChainRun(
            span=span, streamed_token_count=0, run_type=run_type
        )
        self._debug_log("start", span, run_id=run_id)

    def _apply_model_attributes(
        self,
        span: Span,
        serialized: dict[str, Any],
        extra_params: dict[str, Any],
    ) -> None:
        """Attach model/provider attributes from LangChain serialization data."""
        invocation_params = self._get_record(extra_params, "invocation_params") or {}
        serialized_id = self._serialized_id(serialized)
        model_name = (
            self._get_string(invocation_params, "model")
            or self._get_string(invocation_params, "modelName")
            or self._get_string(invocation_params, "model_name")
            or self._get_string(serialized, "name")
            or serialized_id
        )
        provider = (
            self._get_string(invocation_params, "model_provider")
            or self._get_string(invocation_params, "provider")
            or self._infer_provider(serialized_id, model_name)
        )
        span.set_attribute("gen_ai.system", provider)
        span.set_attribute("gen_ai.request.model", model_name)
        span.set_attribute("llm.model_name", model_name)
        span.set_attribute("metadata.ls_model_name", model_name)
        span.set_attribute("langchain.serialized.id", serialized_id)

    def _apply_trace_context(self, span: Span, metadata: dict[str, Any] | None) -> None:
        """Attach Moduna and LangSmith-compatible trace context attributes."""
        conversation_id = (
            self._get_string(metadata, "conversationId")
            or self._get_string(metadata, "conversation_id")
            or self._get_string(metadata, "moduna.conversation.id")
            or self.trace_context.get("conversation_id")
            or self.trace_context.get("conversationId")
        )
        session_id = (
            self._get_string(metadata, "sessionId")
            or self._get_string(metadata, "session_id")
            or self._get_string(metadata, "moduna.session.id")
            or self.trace_context.get("session_id")
            or self.trace_context.get("sessionId")
        )

        if conversation_id:
            span.set_attribute("moduna.conversation.id", conversation_id)
            span.set_attribute("langsmith.metadata.conversation_id", conversation_id)
        if session_id:
            span.set_attribute("moduna.session.id", session_id)
            span.set_attribute("langsmith.metadata.session_id", session_id)
            span.set_attribute("langsmith.trace.session_id", session_id)

    def _apply_prompt_attributes(self, span: Span, messages: list[dict[str, str]]) -> None:
        """Attach prompt inputs in GenAI semantic attribute format."""
        if not messages:
            return
        for index, message in enumerate(messages):
            span.set_attribute(f"gen_ai.prompt.{index}.role", message["role"])
            span.set_attribute(f"gen_ai.prompt.{index}.content", message["content"])
        span.set_attribute("gen_ai.input.messages", json.dumps(messages))
        span.add_event("gen_ai.content.prompt", {"content": json.dumps(messages)})

    def _apply_invocation_attributes(self, span: Span, extra_params: dict[str, Any]) -> None:
        """Attach provider invocation parameters when OpenTelemetry supports them."""
        invocation_params = self._get_record(extra_params, "invocation_params")
        if not invocation_params:
            return

        mappings = {
            "temperature": "gen_ai.request.temperature",
            "top_p": "gen_ai.request.top_p",
            "max_tokens": "gen_ai.request.max_tokens",
            "maxOutputTokens": "gen_ai.request.max_tokens",
            "frequency_penalty": "gen_ai.request.frequency_penalty",
            "presence_penalty": "gen_ai.request.presence_penalty",
            "seed": "gen_ai.request.seed",
            "stop": "gen_ai.request.stop_sequences",
            "stop_sequences": "gen_ai.request.stop_sequences",
            "top_k": "gen_ai.request.top_k",
            "encoding_formats": "gen_ai.request.encoding_formats",
            "tools": "tools",
        }
        for source_key, attribute_key in mappings.items():
            self._set_attribute_if_supported(span, attribute_key, invocation_params.get(source_key))
        span.set_attribute("llm.invocation_parameters", json.dumps(invocation_params))

    def _apply_completion_attributes(self, span: Span, response: Any) -> None:
        """Attach completion outputs in GenAI semantic attribute format."""
        messages: list[dict[str, str]] = []
        for generation_group in self._get_value(response, "generations", []) or []:
            for generation in generation_group:
                messages.append(self._generation_message(generation))

        for index, message in enumerate(messages):
            span.set_attribute(f"gen_ai.completion.{index}.role", message["role"])
            span.set_attribute(f"gen_ai.completion.{index}.content", message["content"])
        if messages:
            span.set_attribute("gen_ai.output.messages", json.dumps(messages))
            span.add_event("gen_ai.content.completion", {"content": json.dumps(messages)})

        response_model = self._response_model(response)
        if response_model:
            span.set_attribute("gen_ai.response.model", response_model)

    def _apply_usage_attributes(self, span: Span, response: Any, streamed_token_count: int) -> None:
        """Attach token usage attributes from LangChain response metadata."""
        usage = self._extract_usage(response)
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens") or (
            streamed_token_count if streamed_token_count > 0 else None
        )
        total_tokens = usage.get("total_tokens")
        reasoning_tokens = usage.get("reasoning_tokens")

        if prompt_tokens is not None:
            span.set_attribute("gen_ai.usage.input_tokens", prompt_tokens)
            span.set_attribute("gen_ai.usage.prompt_tokens", prompt_tokens)
            span.set_attribute("llm.token_count.prompt", prompt_tokens)
        if completion_tokens is not None:
            span.set_attribute("gen_ai.usage.output_tokens", completion_tokens)
            span.set_attribute("gen_ai.usage.completion_tokens", completion_tokens)
            span.set_attribute("llm.token_count.completion", completion_tokens)
        if total_tokens is not None:
            span.set_attribute("gen_ai.usage.total_tokens", total_tokens)
            span.set_attribute("llm.token_count.total", total_tokens)
            span.set_attribute("llm.usage.total_tokens", total_tokens)
        if reasoning_tokens is not None:
            span.set_attribute("gen_ai.usage.details.reasoning_tokens", reasoning_tokens)

    def _extract_usage(self, response: Any) -> dict[str, int]:
        """Extract token usage from LangChain generations and llm_output."""
        usage_metadata = self._first_usage_metadata(response) or {}
        llm_output = self._get_record(response, "llm_output") or {}
        token_usage = (
            self._get_record(llm_output, "tokenUsage")
            or self._get_record(llm_output, "token_usage")
            or {}
        )
        estimated = self._get_record(llm_output, "estimatedTokenUsage") or {}
        output_details = self._get_record(usage_metadata, "output_token_details") or {}

        usage: dict[str, int] = {}
        self._assign_first_number(
            usage,
            "completion_tokens",
            usage_metadata,
            "output_tokens",
            token_usage,
            "completionTokens",
            token_usage,
            "completion_tokens",
            estimated,
            "completionTokens",
        )
        self._assign_first_number(
            usage,
            "prompt_tokens",
            usage_metadata,
            "input_tokens",
            token_usage,
            "promptTokens",
            token_usage,
            "prompt_tokens",
            estimated,
            "promptTokens",
        )
        self._assign_first_number(
            usage,
            "total_tokens",
            usage_metadata,
            "total_tokens",
            token_usage,
            "totalTokens",
            token_usage,
            "total_tokens",
            estimated,
            "totalTokens",
        )
        reasoning = self._get_number(output_details, "reasoning")
        if reasoning is not None:
            usage["reasoning_tokens"] = reasoning
        return usage

    def _first_usage_metadata(self, response: Any) -> dict[str, Any] | None:
        """Find first usage_metadata object on a LangChain generation message."""
        for generation_group in self._get_value(response, "generations", []) or []:
            for generation in generation_group:
                message = self._get_record(generation, "message")
                usage = self._get_record(message, "usage_metadata")
                if usage:
                    return usage
        return None

    def _response_model(self, response: Any) -> str | None:
        """Find provider response model name when LangChain exposes it."""
        llm_output = self._get_record(response, "llm_output") or {}
        for generation_group in self._get_value(response, "generations", []) or []:
            for generation in generation_group:
                message = self._get_record(generation, "message")
                metadata = self._get_record(message, "response_metadata")
                model = self._get_string(metadata, "model_name") or self._get_string(
                    metadata, "model"
                )
                if model:
                    return model
        return self._get_string(llm_output, "model")

    def _normalize_message(self, message: Any) -> dict[str, str]:
        """Convert a LangChain message object into role/content strings."""
        content = self._get_value(message, "content", "")
        message_type = self._get_value(message, "type", "user")
        return {
            "content": content if isinstance(content, str) else json.dumps(content),
            "role": self._map_message_role(str(message_type)),
        }

    def _generation_message(self, generation: Any) -> dict[str, str]:
        """Extract a role/content pair from a LangChain generation candidate."""
        message = self._get_record(generation, "message")
        if message:
            content = self._get_value(message, "content", "")
            message_type = self._get_value(message, "type", "ai")
            return {
                "content": content if isinstance(content, str) else json.dumps(content),
                "role": self._map_message_role(str(message_type)),
            }
        text = self._get_value(generation, "text", "")
        return {"content": str(text), "role": "assistant"}

    def _map_message_role(self, message_type: str) -> str:
        """Map LangChain message types to OpenAI/GenAI role names."""
        return {"ai": "assistant", "human": "user", "system": "system", "tool": "tool"}.get(
            message_type,
            message_type,
        )

    def _infer_provider(self, serialized_id: str, model_name: str) -> str:
        """Infer provider name from serialized LangChain ID and model name."""
        value = f"{serialized_id} {model_name}".lower()
        if "google" in value or "gemini" in value:
            return "google"
        if "openai" in value or "gpt" in value:
            return "openai"
        if "anthropic" in value or "claude" in value:
            return "anthropic"
        return serialized_id.split(".")[-1] if serialized_id else "unknown"

    def _set_attribute_if_supported(self, span: Span, key: str, value: Any) -> None:
        """Set an attribute after converting objects/lists to supported values."""
        if value is None:
            return
        if isinstance(value, str | int | float | bool):
            span.set_attribute(key, value)
            return
        if isinstance(value, list) and all(
            isinstance(item, str | int | float | bool) for item in value
        ):
            span.set_attribute(key, value)
            return
        span.set_attribute(key, json.dumps(value))

    def _serialized_id(self, serialized: dict[str, Any]) -> str:
        """Return LangChain serialized ID as a dot-separated string."""
        value = serialized.get("id", [])
        if isinstance(value, list):
            return ".".join(str(part) for part in value)
        return str(value)

    def _count_generations(self, generations: list[Any]) -> int:
        """Count all candidate generations in a LangChain response."""
        return sum(len(group) for group in generations)

    def _assign_first_number(self, target: dict[str, int], target_key: str, *pairs: Any) -> None:
        """Assign the first available numeric value from record/key pairs."""
        for index in range(0, len(pairs), 2):
            value = self._get_number(pairs[index], pairs[index + 1])
            if value is not None:
                target[target_key] = value
                return

    def _get_record(self, value: Any, key: str) -> dict[str, Any] | None:
        """Read a dict-valued property from dict-like or object-like input."""
        result = self._get_value(value, key)
        return result if isinstance(result, dict) else None

    def _get_string(self, value: Any, key: str) -> str | None:
        """Read a string property from dict-like or object-like input."""
        result = self._get_value(value, key)
        return result if isinstance(result, str) else None

    def _get_number(self, value: Any, key: str) -> int | None:
        """Read an integer property from dict-like or object-like input."""
        result = self._get_value(value, key)
        return result if isinstance(result, int) and not isinstance(result, bool) else None

    def _get_value(self, value: Any, key: str, default: Any = None) -> Any:
        """Read a property from dict-like or object-like input."""
        if isinstance(value, dict):
            return value.get(key, default)
        return getattr(value, key, default)

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


def register_global_moduna_langchain_handler(
    handler: ModunaLangChainCallbackHandler,
    on_failure: Callable[[BaseException], None] | None = None,
) -> None:
    """Best-effort registration of a Moduna handler with Python LangChain globals."""
    try:
        from langchain_core.tracers.context import register_configure_hook, set_tracing_context

        register_configure_hook(handler.name, inheritable=True)
        set_tracing_context(callbacks=[handler])
    except BaseException as exc:
        if on_failure is not None:
            on_failure(exc)
