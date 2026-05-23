"""Typed public data structures for Moduna OpenTelemetry helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

Framework = Literal["langchain"]
"""Framework names supported by the Moduna OpenTelemetry SDK."""


class ModunaTraceContext(TypedDict, total=False):
    """Conversation/session identifiers attached to emitted Moduna spans."""

    conversation_id: str
    session_id: str
    conversationId: str
    sessionId: str


@dataclass(frozen=True, slots=True)
class ModunaOTELConfig:
    """Normalized configuration used to initialize Moduna telemetry."""

    agent_name: str
    framework: Framework
    api_key: str | None = None
    headers: dict[str, str] | None = None
    auto_shutdown: bool = True
