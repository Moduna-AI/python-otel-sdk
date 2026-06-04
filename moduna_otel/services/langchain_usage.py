"""Token usage extraction for LangChain LLM results."""

from __future__ import annotations

from typing import Any

from moduna_otel.services.langchain_utils import (
    get_mapping_or_object,
    get_number,
    get_record,
    get_value,
)


def extract_usage(response: Any) -> dict[str, int]:
    """Extract GenAI token usage metrics from LangChain response metadata."""
    usage_metadata = first_usage_metadata(response) or {}
    llm_output = get_record(response, "llm_output") or {}
    token_usage = (
        get_record(llm_output, "tokenUsage") or get_record(llm_output, "token_usage") or {}
    )
    estimated = get_record(llm_output, "estimatedTokenUsage") or {}
    input_details = get_record(usage_metadata, "input_token_details") or {}
    input_details_alt = get_record(usage_metadata, "input_tokens_details") or {}
    output_details = get_record(usage_metadata, "output_token_details") or {}
    output_details_alt = get_record(usage_metadata, "output_tokens_details") or {}

    usage: dict[str, int] = {}
    _assign_first_number(
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
    _assign_first_number(
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
    _assign_first_number(
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
    reasoning = (
        get_number(usage_metadata, "reasoning_tokens")
        or get_number(output_details, "reasoning")
        or get_number(output_details, "reasoning_tokens")
        or get_number(output_details_alt, "reasoning")
        or get_number(output_details_alt, "reasoning_tokens")
    )
    if reasoning is not None:
        usage["reasoning_tokens"] = reasoning
    cache_read = (
        get_number(usage_metadata, "cache_read_input_tokens")
        or get_number(usage_metadata, "cached_tokens")
        or get_number(input_details, "cache_read")
        or get_number(input_details, "cache_read_tokens")
        or get_number(input_details, "cached_tokens")
        or get_number(input_details_alt, "cache_read")
        or get_number(input_details_alt, "cache_read_tokens")
        or get_number(input_details_alt, "cached_tokens")
    )
    if cache_read is not None:
        usage["cache_read_input_tokens"] = cache_read
    cache_creation = (
        get_number(usage_metadata, "cache_creation_input_tokens")
        or get_number(input_details, "cache_creation")
        or get_number(input_details, "cache_creation_tokens")
        or get_number(input_details_alt, "cache_creation")
        or get_number(input_details_alt, "cache_creation_tokens")
    )
    if cache_creation is not None:
        usage["cache_creation_input_tokens"] = cache_creation
    return usage


def first_usage_metadata(response: Any) -> dict[str, Any] | None:
    """Find first usage_metadata object on a LangChain generation message."""
    for generation_group in get_value(response, "generations", []) or []:
        for generation in generation_group:
            message = get_mapping_or_object(generation, "message")
            usage = get_record(message, "usage_metadata")
            if usage:
                return usage
    return None


def _assign_first_number(target: dict[str, int], target_key: str, *pairs: Any) -> None:
    """Assign the first available numeric value from record/key pairs."""
    for index in range(0, len(pairs), 2):
        value = get_number(pairs[index], pairs[index + 1])
        if value is not None:
            target[target_key] = value
            return
