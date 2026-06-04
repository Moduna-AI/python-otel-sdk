"""GenAI and LangSmith attribute mapping for LangChain spans."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from opentelemetry.trace import Span
from opentelemetry.util.types import AttributeValue

from moduna_otel.services.langchain_usage import extract_usage
from moduna_otel.services.langchain_utils import (
    NormalizedMessage,
    dumps,
    generation_message,
    get_mapping_or_object,
    get_record,
    get_string,
    get_value,
    infer_provider,
    serialized_id,
)
from moduna_otel.types import ModunaTraceContext

GENAI_REQUEST_PARAMETER_ATTRIBUTES: Mapping[str, str] = {
    "temperature": "gen_ai.request.temperature",
    "top_p": "gen_ai.request.top_p",
    "topP": "gen_ai.request.top_p",
    "max_tokens": "gen_ai.request.max_tokens",
    "max_completion_tokens": "gen_ai.request.max_tokens",
    "max_output_tokens": "gen_ai.request.max_tokens",
    "maxOutputTokens": "gen_ai.request.max_tokens",
    "frequency_penalty": "gen_ai.request.frequency_penalty",
    "frequencyPenalty": "gen_ai.request.frequency_penalty",
    "presence_penalty": "gen_ai.request.presence_penalty",
    "presencePenalty": "gen_ai.request.presence_penalty",
    "seed": "gen_ai.request.seed",
    "stop": "gen_ai.request.stop_sequences",
    "stop_sequences": "gen_ai.request.stop_sequences",
    "stopSequences": "gen_ai.request.stop_sequences",
    "top_k": "gen_ai.request.top_k",
    "topK": "gen_ai.request.top_k",
    "encoding_formats": "gen_ai.request.encoding_formats",
}
"""LangChain invocation parameter names mapped to GenAI standard attributes."""

LLM_COMPAT_PARAMETER_ATTRIBUTES: Mapping[str, tuple[str, ...]] = {
    "frequency_penalty": ("llm.frequency_penalty",),
    "frequencyPenalty": ("llm.frequency_penalty",),
    "presence_penalty": ("llm.presence_penalty",),
    "presencePenalty": ("llm.presence_penalty",),
    "functions": ("llm.request.functions",),
    "tools": ("tools",),
    "tool_arguments": ("tool_arguments",),
    "toolArguments": ("tool_arguments",),
}
"""LangSmith-supported compatibility attributes for common LLM tool params."""

GENAI_TOOL_DEFINITION_KEYS: tuple[str, ...] = ("tools", "functions")
"""Invocation parameter keys that carry model request tool definitions."""

TOOL_NAME_KEYS: tuple[str, ...] = ("tool_name", "toolName", "name")
"""Invocation parameter keys LangSmith accepts as ``gen_ai.tool.name``."""


def apply_model_attributes(
    span: Span,
    serialized: dict[str, Any],
    extra_params: dict[str, Any],
) -> None:
    """Attach model/provider attributes from LangChain serialization data."""
    invocation_params = get_record(extra_params, "invocation_params") or {}
    model_id = serialized_id(serialized)
    model_name = (
        get_string(invocation_params, "model")
        or get_string(invocation_params, "modelName")
        or get_string(invocation_params, "model_name")
        or get_string(serialized, "name")
        or model_id
    )
    provider = (
        get_string(invocation_params, "model_provider")
        or get_string(invocation_params, "provider")
        or infer_provider(model_id, model_name)
    )
    span.set_attribute("gen_ai.system", provider)
    span.set_attribute("gen_ai.provider.name", provider)
    span.set_attribute("gen_ai.request.model", model_name)
    span.set_attribute("gen_ai.response.model", model_name)
    span.set_attribute("llm.model_name", model_name)
    span.set_attribute("metadata.ls_provider", provider)
    span.set_attribute("metadata.ls_model_name", model_name)
    span.set_attribute("langchain.serialized.id", model_id)


def apply_trace_context(
    span: Span,
    metadata: dict[str, Any] | None,
    default_context: ModunaTraceContext,
) -> None:
    """Attach Moduna and LangSmith-compatible trace context attributes."""
    conversation_id = (
        get_string(metadata, "conversationId")
        or get_string(metadata, "conversation_id")
        or get_string(metadata, "moduna.conversation.id")
        or default_context.get("conversation_id")
        or default_context.get("conversationId")
    )
    session_id = (
        get_string(metadata, "sessionId")
        or get_string(metadata, "session_id")
        or get_string(metadata, "moduna.session.id")
        or default_context.get("session_id")
        or default_context.get("sessionId")
    )

    if conversation_id:
        span.set_attribute("moduna.conversation.id", conversation_id)
        span.set_attribute("gen_ai.conversation.id", conversation_id)
        span.set_attribute("langsmith.metadata.conversation_id", conversation_id)
    if session_id:
        span.set_attribute("moduna.session.id", session_id)
        span.set_attribute("langsmith.metadata.session_id", session_id)
        span.set_attribute("langsmith.trace.session_id", session_id)
    apply_metadata_attributes(span, metadata)


def apply_metadata_attributes(span: Span, metadata: dict[str, Any] | None) -> None:
    """Attach safe LangSmith metadata attributes from LangChain metadata."""
    if not metadata:
        return

    for key, value in metadata.items():
        if not isinstance(key, str) or key.startswith("moduna."):
            continue
        normalized = metadata_attribute_value(value)
        if normalized is None:
            continue
        attribute_key = (
            key if key.startswith("langsmith.metadata.") else f"langsmith.metadata.{key}"
        )
        span.set_attribute(attribute_key, normalized)


def apply_prompt_attributes(span: Span, messages: list[NormalizedMessage]) -> None:
    """Attach prompt inputs in GenAI semantic attribute format."""
    if not messages:
        return
    for index, message in enumerate(messages):
        span.set_attribute(f"gen_ai.prompt.{index}.role", message["role"])
        span.set_attribute(f"gen_ai.prompt.{index}.content", message["content"])
        span.set_attribute(f"gen_ai.prompt.{index}.message.role", message["role"])
        span.set_attribute(f"gen_ai.prompt.{index}.message.content", message["content"])
    span.set_attribute("gen_ai.prompt", dumps(messages))
    span.set_attribute("gen_ai.input.messages", dumps(messages))
    span.set_attribute("llm.input_messages", dumps(messages))
    span.add_event("gen_ai.content.prompt", {"content": dumps(messages)})


def apply_invocation_attributes(span: Span, extra_params: dict[str, Any]) -> None:
    """Attach provider invocation parameters in LangChain-supported GenAI form."""
    invocation_params = get_record(extra_params, "invocation_params")
    if not invocation_params:
        return

    for source_key, attribute_key in GENAI_REQUEST_PARAMETER_ATTRIBUTES.items():
        set_attribute_if_supported(span, attribute_key, invocation_params.get(source_key))
    for source_key, attribute_keys in LLM_COMPAT_PARAMETER_ATTRIBUTES.items():
        for attribute_key in attribute_keys:
            set_attribute_if_supported(span, attribute_key, invocation_params.get(source_key))
    for source_key in GENAI_TOOL_DEFINITION_KEYS:
        tool_definitions = invocation_params.get(source_key)
        if tool_definitions:
            set_attribute_if_supported(span, "gen_ai.tool.definitions", tool_definitions)
            break
    for source_key in TOOL_NAME_KEYS:
        tool_name = get_string(invocation_params, source_key)
        if tool_name:
            span.set_attribute("gen_ai.tool.name", tool_name)
            break
    else:
        tool_name = first_tool_name(invocation_params)
        if tool_name:
            span.set_attribute("gen_ai.tool.name", tool_name)
    span.set_attribute("llm.invocation_parameters", dumps(invocation_params))


def apply_completion_attributes(span: Span, response: Any) -> None:
    """Attach completion outputs in GenAI semantic attribute format."""
    messages: list[NormalizedMessage] = []
    for generation_group in get_value(response, "generations", []) or []:
        for generation in generation_group:
            messages.append(generation_message(generation))

    for index, message in enumerate(messages):
        span.set_attribute(f"gen_ai.completion.{index}.role", message["role"])
        span.set_attribute(f"gen_ai.completion.{index}.content", message["content"])
        span.set_attribute(f"gen_ai.completion.{index}.message.role", message["role"])
        span.set_attribute(f"gen_ai.completion.{index}.message.content", message["content"])
    if messages:
        span.set_attribute("gen_ai.completion", dumps(messages))
        span.set_attribute("gen_ai.output.messages", dumps(messages))
        span.set_attribute("llm.output_messages", dumps(messages))
        span.add_event("gen_ai.content.completion", {"content": dumps(messages)})

    response_model = response_model_name(response)
    if response_model:
        span.set_attribute("gen_ai.response.model", response_model)
    response_id = response_identifier(response)
    if response_id:
        span.set_attribute("gen_ai.response.id", response_id)
    finish_reasons = response_finish_reasons(response)
    if finish_reasons:
        span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)


def apply_usage_attributes(span: Span, response: Any, streamed_token_count: int) -> None:
    """Attach GenAI usage metrics from LangChain result metadata."""
    usage = extract_usage(response)
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
        span.set_attribute("gen_ai.usage.reasoning.output_tokens", reasoning_tokens)
    cache_read_tokens = usage.get("cache_read_input_tokens")
    if cache_read_tokens is not None:
        span.set_attribute("gen_ai.usage.cache_read.input_tokens", cache_read_tokens)
    cache_creation_tokens = usage.get("cache_creation_input_tokens")
    if cache_creation_tokens is not None:
        span.set_attribute(
            "gen_ai.usage.cache_creation.input_tokens",
            cache_creation_tokens,
        )


def apply_tool_call_attributes(
    span: Span,
    tool_name: str,
    tool_input: Any,
    metadata: dict[str, Any] | None,
    extra_params: dict[str, Any],
) -> None:
    """Attach GenAI tool execution request attributes."""
    span.set_attribute("gen_ai.tool.name", tool_name)
    span.set_attribute("gen_ai.tool.type", "function")
    call_id = (
        get_string(extra_params, "tool_call_id")
        or get_string(extra_params, "toolCallId")
        or get_string(extra_params, "id")
        or get_string(metadata, "tool_call_id")
        or get_string(metadata, "toolCallId")
    )
    if call_id:
        span.set_attribute("gen_ai.tool.call.id", call_id)
    set_attribute_if_supported(span, "gen_ai.tool.call.arguments", tool_input)


def apply_tool_result_attributes(span: Span, output: Any) -> None:
    """Attach GenAI tool execution result attributes."""
    set_attribute_if_supported(span, "gen_ai.tool.call.result", output)


def response_model_name(response: Any) -> str | None:
    """Find provider response model name when LangChain exposes it."""
    llm_output = get_record(response, "llm_output") or {}
    for generation_group in get_value(response, "generations", []) or []:
        for generation in generation_group:
            message = get_mapping_or_object(generation, "message")
            metadata = get_record(message, "response_metadata")
            model = get_string(metadata, "model_name") or get_string(metadata, "model")
            if model:
                return model
    return get_string(llm_output, "model")


def response_identifier(response: Any) -> str | None:
    """Find provider response identifier when exposed by LangChain metadata."""
    llm_output = get_record(response, "llm_output") or {}
    for generation_group in get_value(response, "generations", []) or []:
        for generation in generation_group:
            message = get_mapping_or_object(generation, "message")
            metadata = get_record(message, "response_metadata") or {}
            response_id = (
                get_string(metadata, "id")
                or get_string(metadata, "response_id")
                or get_string(metadata, "system_fingerprint")
            )
            if response_id:
                return response_id
    return get_string(llm_output, "id") or get_string(llm_output, "response_id")


def response_finish_reasons(response: Any) -> list[str] | None:
    """Collect provider finish reasons from LangChain generation metadata."""
    finish_reasons: list[str] = []
    for generation_group in get_value(response, "generations", []) or []:
        for generation in generation_group:
            reason = generation_finish_reason(generation)
            if reason is not None:
                finish_reasons.append(reason)
    return finish_reasons or None


def generation_finish_reason(generation: Any) -> str | None:
    """Find a finish reason on a LangChain generation or its message metadata."""
    direct_reason = get_string(generation, "finish_reason")
    if direct_reason:
        return direct_reason
    generation_info = get_record(generation, "generation_info") or {}
    info_reason = get_string(generation_info, "finish_reason")
    if info_reason:
        return info_reason
    message = get_mapping_or_object(generation, "message")
    metadata = get_record(message, "response_metadata") or {}
    return (
        get_string(metadata, "finish_reason")
        or get_string(metadata, "stop_reason")
        or get_string(metadata, "finishReason")
    )


def first_tool_name(invocation_params: dict[str, Any]) -> str | None:
    """Find the first tool name from OpenAI or Google Gemini tool schemas."""
    tools = get_value(invocation_params, "tools", [])
    if not isinstance(tools, list):
        return None

    for tool in tools:
        if not isinstance(tool, dict):
            continue
        direct_name = get_string(tool, "name")
        if direct_name:
            return direct_name
        function = get_record(tool, "function")
        function_name = get_string(function, "name")
        if function_name:
            return function_name
        declarations = get_value(tool, "functionDeclarations", [])
        if isinstance(declarations, list):
            for declaration in declarations:
                if isinstance(declaration, dict):
                    declaration_name = get_string(declaration, "name")
                    if declaration_name:
                        return declaration_name
    return None


def set_attribute_if_supported(span: Span, key: str, value: Any) -> None:
    """Set an attribute after converting objects/lists to supported values."""
    normalized = span_attribute_value(value)
    if normalized is None:
        return
    span.set_attribute(key, normalized)


def span_attribute_value(value: Any) -> AttributeValue | None:
    """Return a value supported by OpenTelemetry span attributes."""
    if value is None:
        return None
    if isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, list | tuple):
        sequence_value = sequence_attribute_value(value)
        if sequence_value is not None:
            return sequence_value
        return dumps(value)
    if isinstance(value, dict | list | tuple):
        return dumps(value)
    return str(value)


def metadata_attribute_value(value: Any) -> AttributeValue | None:
    """Return metadata only when it already fits OpenTelemetry attribute types."""
    if isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, list | tuple):
        return sequence_attribute_value(value)
    return None


def sequence_attribute_value(value: list[Any] | tuple[Any, ...]) -> AttributeValue | None:
    """Return a homogeneous primitive sequence supported by OpenTelemetry."""
    items = list(value)
    if all(isinstance(item, str) for item in items):
        return [item for item in items if isinstance(item, str)]
    if all(isinstance(item, bool) for item in items):
        return [item for item in items if isinstance(item, bool)]
    if all(isinstance(item, int) and not isinstance(item, bool) for item in items):
        return [item for item in items if isinstance(item, int) and not isinstance(item, bool)]
    if all(isinstance(item, int | float) and not isinstance(item, bool) for item in items):
        return [float(item) for item in items if isinstance(item, int | float)]
    return None
