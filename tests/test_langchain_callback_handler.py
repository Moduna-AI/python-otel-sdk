"""Unit tests for Moduna's LangChain callback handler."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from moduna_otel.services import moduna_langchain_callback_handler as module
from moduna_otel.services.moduna_langchain_callback_handler import (
    ModunaAsyncLangChainCallbackHandler,
    ModunaLangChainCallbackHandler,
)


class FakeSpan:
    """Minimal span fake for LangChain handler assertions."""

    def __init__(self) -> None:
        """Create empty observation containers."""
        self.attributes: dict[str, object] = {}
        self.events: list[tuple[str, dict[str, object]]] = []
        self.exceptions: list[BaseException] = []
        self.ended = False

    def set_attribute(self, key: str, value: object) -> None:
        """Capture an attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, object]) -> None:
        """Capture an event."""
        self.events.append((name, attributes))

    def record_exception(self, exc: BaseException) -> None:
        """Capture an exception."""
        self.exceptions.append(exc)

    def set_status(self, status: object) -> None:
        """Capture span status."""
        self.status = status

    def end(self) -> None:
        """Mark the span ended."""
        self.ended = True

    def get_span_context(self) -> SimpleNamespace:
        """Return a fake context for debug logging."""
        return SimpleNamespace(trace_id=1, span_id=2)


class FakeTracer:
    """Tracer fake that returns a single span."""

    def __init__(self, span: FakeSpan) -> None:
        """Store the returned span."""
        self.span = span

    def start_span(self, name: str, kind: object) -> FakeSpan:
        """Capture start arguments and return the span."""
        self.name = name
        self.kind = kind
        return self.span


class FakeSequenceTracer:
    """Tracer fake that returns spans in creation order."""

    def __init__(self, spans: list[FakeSpan]) -> None:
        """Store the spans returned by start_span calls."""
        self.spans = spans
        self.names: list[str] = []

    def start_span(self, name: str, kind: object) -> FakeSpan:
        """Capture start arguments and return the next span."""
        self.names.append(name)
        return self.spans[len(self.names) - 1]


def test_llm_run_records_prompt_completion_usage_and_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM callbacks populate the core Moduna, LangChain, GenAI attributes."""
    span = FakeSpan()
    monkeypatch.setattr(module.trace, "get_tracer", Mock(return_value=FakeTracer(span)))
    handler = ModunaLangChainCallbackHandler({"conversation_id": "c1", "session_id": "s1"})

    handler.on_llm_start(
        {"id": ["langchain", "chat_models", "openai"], "name": "gpt-4o-mini"},
        ["hello"],
        run_id="run-1",
        tags=["prod"],
        invocation_params={
            "temperature": 0.2,
            "model": "gpt-4o-mini",
            "top_p": 0.8,
            "max_tokens": 64,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
            "seed": 7,
            "stop": ["END"],
            "top_k": 3,
            "encoding_formats": ["float"],
            "tools": [{"name": "lookup"}],
            "tool_name": "lookup",
            "tool_arguments": {"query": "status"},
        },
    )
    handler.on_llm_new_token("hi", run_id="run-1")
    handler.on_llm_end(
        {
            "generations": [
                [
                    {
                        "message": {
                            "content": "hi",
                            "type": "ai",
                            "response_metadata": {
                                "id": "resp-1",
                                "model_name": "gpt-4o-mini-2026",
                                "finish_reason": "stop",
                            },
                            "usage_metadata": {
                                "input_tokens": 3,
                                "output_tokens": 2,
                                "total_tokens": 5,
                                "reasoning_tokens": 1,
                                "input_token_details": {
                                    "cache_read": 2,
                                    "cache_creation": 4,
                                },
                            },
                        }
                    }
                ]
            ],
            "llm_output": {"tokenUsage": {"promptTokens": 3, "totalTokens": 5}},
        },
        run_id="run-1",
    )

    assert span.attributes["moduna.framework"] == "langchain"
    assert span.attributes["moduna.conversation.id"] == "c1"
    assert span.attributes["gen_ai.conversation.id"] == "c1"
    assert span.attributes["moduna.session.id"] == "s1"
    assert span.attributes["gen_ai.provider.name"] == "openai"
    assert span.attributes["gen_ai.prompt.0.content"] == "hello"
    assert span.attributes["gen_ai.prompt.0.message.content"] == "hello"
    assert json.loads(str(span.attributes["gen_ai.prompt"])) == [
        {"role": "user", "content": "hello"}
    ]
    assert json.loads(str(span.attributes["gen_ai.input.messages"])) == [
        {"role": "user", "content": "hello"}
    ]
    assert span.attributes["gen_ai.completion.0.content"] == "hi"
    assert span.attributes["gen_ai.completion.0.message.role"] == "assistant"
    assert json.loads(str(span.attributes["gen_ai.completion"])) == [
        {"role": "assistant", "content": "hi"}
    ]
    assert span.attributes["gen_ai.request.temperature"] == 0.2
    assert span.attributes["gen_ai.request.top_p"] == 0.8
    assert span.attributes["gen_ai.request.max_tokens"] == 64
    assert span.attributes["gen_ai.request.frequency_penalty"] == 0.1
    assert span.attributes["gen_ai.request.presence_penalty"] == 0.2
    assert span.attributes["gen_ai.request.seed"] == 7
    assert span.attributes["gen_ai.request.stop_sequences"] == ["END"]
    assert span.attributes["gen_ai.request.top_k"] == 3
    assert span.attributes["gen_ai.request.encoding_formats"] == ["float"]
    assert span.attributes["gen_ai.tool.name"] == "lookup"
    assert json.loads(str(span.attributes["gen_ai.tool.definitions"])) == [{"name": "lookup"}]
    assert json.loads(str(span.attributes["tools"])) == [{"name": "lookup"}]
    assert json.loads(str(span.attributes["tool_arguments"])) == {"query": "status"}
    assert span.attributes["llm.frequency_penalty"] == 0.1
    assert span.attributes["llm.presence_penalty"] == 0.2
    assert span.attributes["gen_ai.response.model"] == "gpt-4o-mini-2026"
    assert span.attributes["gen_ai.response.id"] == "resp-1"
    assert span.attributes["gen_ai.response.finish_reasons"] == ["stop"]
    assert span.attributes["gen_ai.usage.prompt_tokens"] == 3
    assert span.attributes["gen_ai.usage.input_tokens"] == 3
    assert span.attributes["gen_ai.usage.completion_tokens"] == 2
    assert span.attributes["gen_ai.usage.output_tokens"] == 2
    assert span.attributes["gen_ai.usage.total_tokens"] == 5
    assert span.attributes["gen_ai.usage.details.reasoning_tokens"] == 1
    assert span.attributes["gen_ai.usage.reasoning.output_tokens"] == 1
    assert span.attributes["gen_ai.usage.cache_read.input_tokens"] == 2
    assert span.attributes["gen_ai.usage.cache_creation.input_tokens"] == 4
    assert span.attributes["llm.token_count.prompt"] == 3
    assert span.attributes["llm.token_count.completion"] == 2
    assert span.attributes["llm.token_count.total"] == 5
    assert span.ended is True
    assert "run-1" not in handler.runs


def test_tool_run_records_execution_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tool callbacks populate GenAI execute_tool span attributes."""
    span = FakeSpan()
    monkeypatch.setattr(module.trace, "get_tracer", Mock(return_value=FakeTracer(span)))
    handler = ModunaLangChainCallbackHandler({"conversation_id": "c1", "session_id": "s1"})

    handler.on_tool_start(
        {"id": ["langchain", "tools", "lookup"], "name": "lookup"},
        '{"query":"status"}',
        run_id="tool-run-1",
        parent_run_id="run-1",
        tags=["prod"],
        metadata={"conversation_id": "c2"},
        inputs={"query": "status"},
        tool_call_id="call-1",
    )
    handler.on_tool_end({"status": "ok"}, run_id="tool-run-1")

    assert span.attributes["moduna.framework"] == "langchain"
    assert span.attributes["langchain.run.id"] == "tool-run-1"
    assert span.attributes["langchain.parent_run.id"] == "run-1"
    assert span.attributes["langchain.run.type"] == "tool"
    assert span.attributes["langsmith.span.kind"] == "tool"
    assert span.attributes["gen_ai.operation.name"] == "execute_tool"
    assert span.attributes["gen_ai.tool.name"] == "lookup"
    assert span.attributes["gen_ai.tool.type"] == "function"
    assert span.attributes["gen_ai.tool.call.id"] == "call-1"
    assert json.loads(str(span.attributes["gen_ai.tool.call.arguments"])) == {"query": "status"}
    assert json.loads(str(span.attributes["gen_ai.tool.call.result"])) == {"status": "ok"}
    assert span.attributes["gen_ai.conversation.id"] == "c2"
    assert span.ended is True
    assert "tool-run-1" not in handler.runs


def test_tool_error_records_exception_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tool errors emit standard error attributes and close the active span."""
    span = FakeSpan()
    monkeypatch.setattr(module.trace, "get_tracer", Mock(return_value=FakeTracer(span)))
    handler = ModunaLangChainCallbackHandler()
    error = ValueError("bad tool")

    handler.on_tool_start({"name": "lookup"}, "{}", run_id="tool-run-2")
    handler.on_tool_error(error, run_id="tool-run-2")

    assert span.attributes["gen_ai.operation.name"] == "execute_tool"
    assert span.attributes["error.type"] == "ValueError"
    assert span.attributes["error.message"] == "bad tool"
    assert span.exceptions == [error]
    assert span.ended is True
    assert "tool-run-2" not in handler.runs


def test_chain_run_records_inputs_outputs_and_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chain callbacks populate invoke_agent span attributes."""
    span = FakeSpan()
    monkeypatch.setattr(module.trace, "get_tracer", Mock(return_value=FakeTracer(span)))
    handler = ModunaLangChainCallbackHandler({"conversation_id": "c1"})

    handler.on_chain_start(
        {"id": ["langchain", "chains", "AgentExecutor"], "name": "agent"},
        {"input": "plan this"},
        run_id="chain-run-1",
        parent_run_id="root-run",
        tags=["agent"],
        metadata={"session_id": "s1"},
    )
    handler.on_chain_end({"output": "done"}, run_id="chain-run-1")

    assert span.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert span.attributes["langchain.run.type"] == "chain"
    assert span.attributes["langsmith.span.kind"] == "chain"
    assert span.attributes["langchain.parent_run.id"] == "root-run"
    assert span.attributes["langchain.serialized.id"] == "langchain.chains.AgentExecutor"
    assert json.loads(str(span.attributes["langchain.input"])) == {"input": "plan this"}
    assert json.loads(str(span.attributes["langchain.output"])) == {"output": "done"}
    assert span.attributes["moduna.session.id"] == "s1"
    assert span.ended is True


def test_chain_error_records_standard_error_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chain errors record exception details and close the span."""
    span = FakeSpan()
    monkeypatch.setattr(module.trace, "get_tracer", Mock(return_value=FakeTracer(span)))
    handler = ModunaLangChainCallbackHandler()
    error = RuntimeError("chain failed")

    handler.on_chain_start({"name": "agent"}, {"input": "hello"}, run_id="chain-run-2")
    handler.on_chain_error(error, run_id="chain-run-2")

    assert span.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert span.attributes["error.type"] == "RuntimeError"
    assert span.attributes["error.message"] == "chain failed"
    assert span.exceptions == [error]
    assert span.ended is True


def test_retriever_run_records_query_and_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retriever callbacks populate retrieval span attributes."""
    span = FakeSpan()
    monkeypatch.setattr(module.trace, "get_tracer", Mock(return_value=FakeTracer(span)))
    handler = ModunaLangChainCallbackHandler()
    documents = [
        SimpleNamespace(page_content="doc one", metadata={"source": "a"}),
        {"page_content": "doc two", "metadata": {"source": "b"}},
    ]

    handler.on_retriever_start(
        {"id": ["langchain", "retrievers", "VectorStoreRetriever"]},
        "find docs",
        run_id="retriever-run-1",
        metadata={"conversation_id": "c1"},
    )
    handler.on_retriever_end(documents, run_id="retriever-run-1")

    assert span.attributes["gen_ai.operation.name"] == "retrieval"
    assert span.attributes["langchain.run.type"] == "retriever"
    assert span.attributes["langsmith.span.kind"] == "retriever"
    assert span.attributes["gen_ai.retrieval.query"] == "find docs"
    assert span.attributes["gen_ai.retrieval.documents.count"] == 2
    assert "doc one" in str(span.attributes["gen_ai.retrieval.documents"])
    assert span.attributes["gen_ai.conversation.id"] == "c1"
    assert span.ended is True


def test_retriever_error_records_standard_error_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retriever errors record exception details and close the span."""
    span = FakeSpan()
    monkeypatch.setattr(module.trace, "get_tracer", Mock(return_value=FakeTracer(span)))
    handler = ModunaLangChainCallbackHandler()
    error = LookupError("missing index")

    handler.on_retriever_start({"name": "retriever"}, "query", run_id="retriever-run-2")
    handler.on_retriever_error(error, run_id="retriever-run-2")

    assert span.attributes["gen_ai.operation.name"] == "retrieval"
    assert span.attributes["error.type"] == "LookupError"
    assert span.attributes["error.message"] == "missing index"
    assert span.exceptions == [error]
    assert span.ended is True


def test_async_handler_records_lifecycle_spans_and_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Async callbacks mirror sync tracing and record async-only events."""
    spans = [FakeSpan(), FakeSpan(), FakeSpan(), FakeSpan()]
    tracer = FakeSequenceTracer(spans)
    monkeypatch.setattr(module.trace, "get_tracer", Mock(return_value=tracer))
    handler = ModunaAsyncLangChainCallbackHandler({"session_id": "s1"})

    async def run_callbacks() -> None:
        await handler.on_llm_start(
            {"id": ["langchain", "llms", "openai"], "name": "gpt-4o-mini"},
            ["hello"],
            run_id="async-llm",
            invocation_params={"model": "gpt-4o-mini"},
        )
        await handler.on_llm_new_token("hi", run_id="async-llm")
        await handler.on_llm_end(
            {
                "generations": [[{"text": "hi"}]],
                "llm_output": {"tokenUsage": {"completionTokens": 1}},
            },
            run_id="async-llm",
        )
        await handler.on_tool_start(
            {"name": "lookup"},
            "{}",
            run_id="async-tool",
            inputs={"query": "status"},
        )
        await handler.on_tool_end("ok", run_id="async-tool")
        await handler.on_chain_start(
            {"name": "agent"},
            {"input": "hello"},
            run_id="async-chain",
        )
        await handler.on_text("thinking", run_id="async-chain")
        await handler.on_agent_action(SimpleNamespace(tool="lookup"), run_id="async-chain")
        await handler.on_chain_end({"output": "done"}, run_id="async-chain")
        await handler.on_custom_event(
            "external",
            {"value": 1},
            run_id="async-event",
            metadata={"session_id": "s2"},
        )

    asyncio.run(run_callbacks())

    assert spans[0].attributes["gen_ai.operation.name"] == "text_completion"
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 1
    assert spans[0].ended is True
    assert spans[1].attributes["gen_ai.operation.name"] == "execute_tool"
    assert json.loads(str(spans[1].attributes["gen_ai.tool.call.arguments"])) == {
        "query": "status"
    }
    assert spans[1].attributes["gen_ai.tool.call.result"] == "ok"
    assert spans[1].ended is True
    assert spans[2].attributes["gen_ai.operation.name"] == "invoke_agent"
    assert ("langchain.text", {"text": "thinking"}) in spans[2].events
    assert spans[2].ended is True
    assert spans[3].attributes["langchain.run.type"] == "event"
    assert spans[3].attributes["moduna.session.id"] == "s2"
    assert spans[3].events[0][0] == "langchain.custom.external"
    assert spans[3].ended is True


def test_async_retriever_lifecycle_records_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    """Async retriever callbacks create retrieval spans."""
    span = FakeSpan()
    monkeypatch.setattr(module.trace, "get_tracer", Mock(return_value=FakeTracer(span)))
    handler = ModunaAsyncLangChainCallbackHandler()

    async def run_callbacks() -> None:
        await handler.on_retriever_start(
            {"name": "retriever"},
            "query",
            run_id="async-retriever",
        )
        await handler.on_retriever_end(
            [SimpleNamespace(page_content="answer", metadata={"score": 1})],
            run_id="async-retriever",
        )

    asyncio.run(run_callbacks())

    assert span.attributes["gen_ai.operation.name"] == "retrieval"
    assert span.attributes["gen_ai.retrieval.documents.count"] == 1
    assert "answer" in str(span.attributes["gen_ai.retrieval.documents"])
    assert span.ended is True


def test_chat_model_metadata_overrides_default_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """LangChain metadata can override default conversation and session IDs."""
    span = FakeSpan()
    monkeypatch.setattr(module.trace, "get_tracer", Mock(return_value=FakeTracer(span)))
    handler = ModunaLangChainCallbackHandler({"conversation_id": "default"})
    message = SimpleNamespace(content="hello", type="human")

    handler.on_chat_model_start(
        {"id": ["langchain", "chat_models", "anthropic"]},
        [[message]],
        run_id="run-2",
        metadata={"conversationId": "override", "moduna.session.id": "s2"},
    )

    assert span.attributes["moduna.conversation.id"] == "override"
    assert span.attributes["moduna.session.id"] == "s2"
    assert span.attributes["gen_ai.operation.name"] == "chat"


def test_chat_model_safe_metadata_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    """Primitive LangChain metadata is emitted as LangSmith metadata."""
    span = FakeSpan()
    monkeypatch.setattr(module.trace, "get_tracer", Mock(return_value=FakeTracer(span)))
    handler = ModunaLangChainCallbackHandler()
    message = SimpleNamespace(content="hello", type="human")

    handler.on_chat_model_start(
        {"id": ["langchain", "chat_models", "google"], "name": "gemini"},
        [[message]],
        run_id="run-4",
        metadata={
            "customer_tier": "enterprise",
            "labels": ["priority", "chat"],
            "nested": {"ignored": True},
        },
    )

    assert span.attributes["langsmith.metadata.customer_tier"] == "enterprise"
    assert span.attributes["langsmith.metadata.labels"] == ["priority", "chat"]
    assert "langsmith.metadata.nested" not in span.attributes


def test_google_weather_tool_object_response_records_usage_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Google object-shaped LangChain generations still emit GenAI metrics."""
    span = FakeSpan()
    monkeypatch.setattr(module.trace, "get_tracer", Mock(return_value=FakeTracer(span)))
    handler = ModunaLangChainCallbackHandler(
        {
            "conversation_id": "conversation-real-langchain-weather-tool",
            "session_id": "session-real-langchain-weather-tool",
        }
    )
    user_message = SimpleNamespace(
        content="Use the get_weather tool to get weather information for Chennai, India.",
        type="human",
    )

    handler.on_chat_model_start(
        {"id": ["langchain", "chat_models", "google_genai", "ChatGoogleGenerativeAI"]},
        [[user_message]],
        run_id="run-weather",
        invocation_params={
            "tools": [
                {
                    "functionDeclarations": [
                        {
                            "name": "get_weather",
                            "description": "Get current weather information.",
                            "parameters": {
                                "properties": {"location": {"type": "string"}},
                                "required": ["location"],
                                "type": "object",
                            },
                        }
                    ]
                }
            ],
            "toolConfig": {
                "functionCallingConfig": {
                    "mode": "ANY",
                    "allowedFunctionNames": ["get_weather"],
                }
            },
        },
    )
    handler.on_llm_end(
        SimpleNamespace(
            generations=[
                [
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=[
                                {
                                    "type": "functionCall",
                                    "functionCall": {
                                        "name": "get_weather",
                                        "args": {"location": "Chennai, India"},
                                    },
                                }
                            ],
                            type="ai",
                            response_metadata={
                                "model_name": "gemini-2.5-flash",
                                "model_provider": "google_genai",
                            },
                            usage_metadata={
                                "input_tokens": 68,
                                "output_tokens": 17,
                                "total_tokens": 85,
                            },
                        )
                    )
                ]
            ],
            llm_output={},
        ),
        run_id="run-weather",
    )

    assert span.attributes["metadata.ls_provider"] == "google"
    assert span.attributes["metadata.ls_model_name"] == (
        "langchain.chat_models.google_genai.ChatGoogleGenerativeAI"
    )
    assert span.attributes["gen_ai.response.model"] == "gemini-2.5-flash"
    assert span.attributes["gen_ai.tool.name"] == "get_weather"
    assert "get_weather" in str(span.attributes["tools"])
    assert span.attributes["gen_ai.usage.input_tokens"] == 68
    assert span.attributes["gen_ai.usage.prompt_tokens"] == 68
    assert span.attributes["gen_ai.usage.output_tokens"] == 17
    assert span.attributes["gen_ai.usage.completion_tokens"] == 17
    assert span.attributes["gen_ai.usage.total_tokens"] == 85
    assert span.attributes["llm.token_count.prompt"] == 68
    assert span.attributes["llm.token_count.completion"] == 17
    assert span.attributes["llm.token_count.total"] == 85
    assert "functionCall" in str(span.attributes["gen_ai.completion.0.content"])


def test_llm_error_records_exception_and_ends_span(monkeypatch: pytest.MonkeyPatch) -> None:
    """LangChain error callback records the failure and closes the run span."""
    span = FakeSpan()
    monkeypatch.setattr(module.trace, "get_tracer", Mock(return_value=FakeTracer(span)))
    handler = ModunaLangChainCallbackHandler()
    error = RuntimeError("failed")

    handler.on_llm_start({"id": ["x"]}, ["hello"], run_id="run-3")
    handler.on_llm_error(error, run_id="run-3")

    assert span.exceptions == [error]
    assert span.ended is True
    assert "run-3" not in handler.runs
