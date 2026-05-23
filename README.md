# moduna-otel

Python OpenTelemetry helpers for Moduna AI traces.

`moduna-otel` makes Moduna tracing easy to add with one import and one
instantiation. It configures OpenTelemetry once, exports traces to Moduna over
OTLP HTTP, and provides ergonomic helpers for LangChain and Vercel AI SDK style
telemetry metadata.

## Installation

Install from the built wheel or from PyPI once published:

```bash
pip install moduna-otel
```

For LangChain callback support, install the optional LangChain dependency:

```bash
pip install "moduna-otel[langchain]"
```

When working from this repository, use `uv`:

```bash
uv sync --extra dev --extra langchain
```

## Getting Started

Set your Moduna API key with an environment variable:

```bash
export MODUNA_API_KEY="md_live_..."
```

Create the SDK once near application startup:

```python
from moduna_otel import ModunaOTEL

otel = ModunaOTEL(
    agent_name="support-agent",
    framework="langchain",
)
```

You can also pass the API key directly:

```python
from moduna_otel import ModunaOTEL

otel = ModunaOTEL(
    agent_name="support-agent",
    framework="langchain",
    api_key="md_live_...",
)
```

Multiple `ModunaOTEL(...)` instances share the same OpenTelemetry provider, so
creating per-module wrappers will not create duplicate exporters.

## LangChain

Create a LangChain callback handler and pass it into your model or chain call:

```python
from moduna_otel import ModunaOTEL

otel = ModunaOTEL(
    agent_name="support-agent",
    framework="langchain",
)

handler = otel.langchain_handler(
    {
        "conversation_id": "conversation-123",
        "session_id": "session-456",
    }
)

result = chain.invoke(
    {"input": "Summarize this ticket"},
    config={
        "callbacks": [handler],
        "metadata": {
            "conversationId": "conversation-123",
            "sessionId": "session-456",
        },
    },
)
```

The handler records LLM/chat model spans, prompts, completions, token usage when
available, streamed token events, Moduna conversation/session attributes, and
errors.

For applications that use global LangChain callbacks, registration is available
as a best-effort helper:

```python
otel.register_global_langchain_handler(
    {
        "conversation_id": "conversation-123",
        "session_id": "session-456",
    }
)
```

LangChain global callback APIs vary across versions. If global registration is
not available in your installed LangChain version, Moduna logs one warning and
continues without breaking application code.

## Vercel AI SDK Style Telemetry

The Python package provides metadata compatible with Moduna's Vercel AI SDK
telemetry keys:

```python
from moduna_otel import ModunaOTEL

otel = ModunaOTEL(
    agent_name="support-agent",
    framework="vercel-ai-sdk",
)

telemetry = otel.vercel_telemetry(
    {
        "conversation_id": "conversation-123",
        "session_id": "session-456",
    }
)

assert telemetry == {
    "isEnabled": True,
    "metadata": {
        "moduna.conversation.id": "conversation-123",
        "moduna.session.id": "session-456",
    },
}
```

Use this dictionary anywhere your integration accepts metadata-style telemetry,
or forward the `metadata` values to a TypeScript service using Vercel AI SDK
`experimental_telemetry`.

## Manual Instrumentation

Use `instrument()` to wrap custom model calls or application work in a Moduna
CLIENT span:

```python
from moduna_otel import ModunaOTEL

otel = ModunaOTEL(agent_name="support-agent", framework="langchain")


def call_model(span):
    span.set_attribute("app.operation", "ticket-summary")
    return model.invoke("Summarize this ticket")


result = otel.instrument(
    "support-agent.summary",
    call_model,
    {
        "conversation_id": "conversation-123",
        "session_id": "session-456",
    },
)
```

If the callback raises an exception, the exception is recorded on the span and
then re-raised.

## Configuration

`ModunaOTEL` accepts keyword arguments:

```python
ModunaOTEL(
    agent_name="support-agent",
    framework="langchain",  # "langchain" or "vercel-ai-sdk"
    api_key=None,           # defaults to MODUNA_API_KEY
    headers=None,           # extra OTLP HTTP headers
    auto_shutdown=True,     # flush on process exit
)
```

It also accepts a mapping:

```python
otel = ModunaOTEL(
    {
        "agentName": "support-agent",
        "framework": "langchain",
        "autoShutdown": True,
    }
)
```

## Development

Install development dependencies:

```bash
uv sync --extra dev --extra langchain
```

Run tests:

```bash
uv run pytest
```

Format code:

```bash
uv run ruff format
```

Run the linter:

```bash
uv run ruff check
```

Build source and wheel distributions:

```bash
uv build
```

The built artifacts are written to `dist/`.

## Contributing

Contributions are welcome. Please keep changes focused and aligned with the
TypeScript SDK behavior.

Before opening a pull request:

1. Add or update focused pytest coverage for behavior changes.
2. Run `uv run pytest`.
3. Run `uv run ruff format` and `uv run ruff check`.
4. Run `uv build` for packaging changes.
5. Keep telemetry failures non-fatal to user application code.
6. Avoid broad refactors unless they are required for the change.

Useful areas for contribution include LangChain version compatibility,
additional token-usage extraction cases, and documentation examples for common
framework integrations.
