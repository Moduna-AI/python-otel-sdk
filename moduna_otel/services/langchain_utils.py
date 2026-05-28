"""Utility helpers for LangChain callback telemetry."""

from __future__ import annotations

import json
from typing import Any

NormalizedMessage = dict[str, str]


def get_value(value: Any, key: str, default: Any = None) -> Any:
    """Read a property from dict-like or object-like input."""
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def get_record(value: Any, key: str) -> dict[str, Any] | None:
    """Read a dict-valued property from dict-like or object-like input."""
    result = get_value(value, key)
    return result if isinstance(result, dict) else None


def get_mapping_or_object(value: Any, key: str) -> Any | None:
    """Read a property that may be either a dict or a typed LangChain object."""
    result = get_value(value, key)
    return result if result is not None else None


def get_string(value: Any, key: str) -> str | None:
    """Read a string property from dict-like or object-like input."""
    result = get_value(value, key)
    return result if isinstance(result, str) else None


def get_number(value: Any, key: str) -> int | None:
    """Read an integer property from dict-like or object-like input."""
    result = get_value(value, key)
    return result if isinstance(result, int) and not isinstance(result, bool) else None


def dumps(value: Any) -> str:
    """Serialize telemetry payloads consistently."""
    return json.dumps(value)


def normalize_message(message: Any) -> NormalizedMessage:
    """Convert a LangChain message object into role/content strings."""
    content = get_value(message, "content", "")
    message_type = get_value(message, "type", "user")
    return {
        "content": content if isinstance(content, str) else dumps(content),
        "role": map_message_role(str(message_type)),
    }


def generation_message(generation: Any) -> NormalizedMessage:
    """Extract a role/content pair from a LangChain generation candidate."""
    message = get_mapping_or_object(generation, "message")
    if message is not None:
        content = get_value(message, "content", "")
        message_type = get_value(message, "type", "ai")
        return {
            "content": content if isinstance(content, str) else dumps(content),
            "role": map_message_role(str(message_type)),
        }
    return {"content": str(get_value(generation, "text", "")), "role": "assistant"}


def map_message_role(message_type: str) -> str:
    """Map LangChain message types to OpenAI/GenAI role names."""
    return {"ai": "assistant", "human": "user", "system": "system", "tool": "tool"}.get(
        message_type,
        message_type,
    )


def serialized_id(serialized: dict[str, Any]) -> str:
    """Return LangChain serialized ID as a dot-separated string."""
    value = serialized.get("id", [])
    if isinstance(value, list):
        return ".".join(str(part) for part in value)
    return str(value)


def infer_provider(serialized_model_id: str, model_name: str) -> str:
    """Infer provider name from serialized LangChain ID and model name."""
    value = f"{serialized_model_id} {model_name}".lower()
    if "google" in value or "gemini" in value:
        return "google"
    if "openai" in value or "gpt" in value:
        return "openai"
    if "anthropic" in value or "claude" in value:
        return "anthropic"
    return serialized_model_id.split(".")[-1] if serialized_model_id else "unknown"


def count_generations(generations: list[Any]) -> int:
    """Count all candidate generations in a LangChain response."""
    return sum(len(group) for group in generations)
