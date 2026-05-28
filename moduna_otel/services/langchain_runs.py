"""Run state models and identity attributes for LangChain spans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from opentelemetry.trace import Span


@dataclass(slots=True)
class ActiveLangChainRun:
    """OpenTelemetry state for an active LangChain LLM or chat model run."""

    span: Span
    streamed_token_count: int
    run_type: str


@dataclass(frozen=True, slots=True)
class LangChainRunStart:
    """Normalized inputs needed to start a LangChain model span."""

    serialized: dict[str, Any]
    input_messages: list[dict[str, str]]
    input_count: int
    run_id: str
    parent_run_id: str | None
    tags: list[str] | None
    metadata: dict[str, Any] | None
    run_name: str | None
    run_type: str
    extra_params: dict[str, Any]


def apply_run_attributes(span: Span, run: LangChainRunStart) -> None:
    """Attach LangChain run identity and GenAI operation attributes."""
    operation = "chat" if run.run_type == "chat_model" else "completion"
    span.set_attribute("moduna.framework", "langchain")
    span.set_attribute("sdk.integration", "langchain")
    span.set_attribute("langchain.run.id", run.run_id)
    span.set_attribute("langchain.run.type", run.run_type)
    span.set_attribute("langchain.input.count", run.input_count)
    span.set_attribute("langsmith.span.kind", "llm")
    span.set_attribute("gen_ai.operation.name", operation)
    span.set_attribute("llm.request.type", operation)
    if run.parent_run_id:
        span.set_attribute("langchain.parent_run.id", run.parent_run_id)
    if run.run_name:
        span.set_attribute("langsmith.trace.name", run.run_name)
    if run.tags:
        joined_tags = ",".join(run.tags)
        span.set_attribute("langchain.tags", joined_tags)
        span.set_attribute("langsmith.span.tags", joined_tags)
