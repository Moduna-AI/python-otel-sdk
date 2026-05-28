"""Configuration helpers for Moduna OpenTelemetry setup."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeGuard

from opentelemetry.sdk.trace import TracerProvider

from moduna_otel.types import Framework, ModunaOTELConfig, ModunaTraceContext

DEFAULT_ENDPOINT = "https://volex-otel-git-506013021984.us-central1.run.app/v1/traces"


@dataclass(slots=True)
class SharedOTELState:
    """Mutable singleton state shared by all ModunaOTEL instances."""

    provider: TracerProvider | None = None
    started: bool = False
    shutdown_started: bool = False
    lifecycle_hooks_registered: bool = False
    warned: bool = False


def is_framework(value: object) -> TypeGuard[Framework]:
    """Return whether a value is a supported Python integration framework."""
    return value == "langchain"


def require_framework(value: object) -> Framework:
    """Validate and return a supported Python framework name."""
    if is_framework(value):
        return value
    raise TypeError("ModunaOTEL only supports the 'langchain' framework in Python.")


def normalize_config(
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
            framework=require_framework(agent_name.framework),
            api_key=agent_name.api_key or os.getenv("MODUNA_API_KEY"),
            headers=agent_name.headers,
            auto_shutdown=agent_name.auto_shutdown,
        )

    if isinstance(agent_name, Mapping):
        return _normalize_mapping_config(agent_name, framework, api_key, headers, auto_shutdown)

    if framework is None:
        raise TypeError("ModunaOTEL requires a framework.")

    return ModunaOTELConfig(
        agent_name=agent_name,
        framework=require_framework(framework),
        api_key=api_key or os.getenv("MODUNA_API_KEY"),
        headers=headers,
        auto_shutdown=auto_shutdown,
    )


def export_headers(config: ModunaOTELConfig) -> dict[str, str]:
    """Build OTLP HTTP headers, including Authorization when configured."""
    headers: dict[str, str] = {}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    if config.headers:
        headers.update(config.headers)
    return headers


def trace_metadata(context: ModunaTraceContext) -> dict[str, str]:
    """Convert Moduna context aliases into OpenTelemetry metadata keys."""
    metadata: dict[str, str] = {}
    conversation_id = context.get("conversation_id") or context.get("conversationId")
    session_id = context.get("session_id") or context.get("sessionId")
    if conversation_id:
        metadata["moduna.conversation.id"] = conversation_id
    if session_id:
        metadata["moduna.session.id"] = session_id
    return metadata


def _normalize_mapping_config(
    config: Mapping[str, Any],
    framework: Framework | None,
    api_key: str | None,
    headers: dict[str, str] | None,
    auto_shutdown: bool,
) -> ModunaOTELConfig:
    """Normalize dict-style constructor input."""
    mapped_agent_name = config.get("agent_name") or config.get("agentName")
    if not isinstance(mapped_agent_name, str):
        raise TypeError("ModunaOTEL config requires agent_name.")

    mapped_headers = config.get("headers", headers)
    if mapped_headers is not None and not _is_string_dict(mapped_headers):
        raise TypeError("ModunaOTEL headers must be a dict[str, str].")

    return ModunaOTELConfig(
        agent_name=mapped_agent_name,
        framework=require_framework(config.get("framework", framework)),
        api_key=_optional_string(config.get("api_key") or config.get("apiKey") or api_key)
        or os.getenv("MODUNA_API_KEY"),
        headers=mapped_headers,
        auto_shutdown=bool(config.get("auto_shutdown", config.get("autoShutdown", auto_shutdown))),
    )


def _optional_string(value: object) -> str | None:
    """Return a value only when it is a string."""
    return value if isinstance(value, str) else None


def _is_string_dict(value: object) -> TypeGuard[dict[str, str]]:
    """Return whether a value is a string-to-string dictionary."""
    return isinstance(value, dict) and all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    )
