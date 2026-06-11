# Moduna Python SDK

[![PyPI version](https://img.shields.io/pypi/v/moduna.svg)](https://pypi.org/project/moduna/)
[![Python versions](https://img.shields.io/pypi/pyversions/moduna.svg)](https://pypi.org/project/moduna/)
[![Package status](https://img.shields.io/pypi/status/moduna.svg)](https://pypi.org/project/moduna/)
[![License](https://img.shields.io/pypi/l/moduna.svg)](https://github.com/Moduna-AI/python-otel-sdk/blob/main/pyproject.toml)
[![GitHub release](https://img.shields.io/github/v/release/Moduna-AI/python-otel-sdk.svg)](https://github.com/Moduna-AI/python-otel-sdk/releases)
[![Upload Python Package](https://github.com/Moduna-AI/python-otel-sdk/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Moduna-AI/python-otel-sdk/actions/workflows/python-publish.yml)
[![Downloads](https://img.shields.io/pypi/dm/moduna.svg)](https://pypi.org/project/moduna/)

Moduna is a small Python SDK that brands and simplifies Traceloop setup for AI agent frameworks. It gives
applications a stable `moduna` import path, framework-aware instrumentation, and Moduna API-key resolution from
configuration, exported environment variables, or `.env`.

## Installation

Install the SDK with all supported framework integrations:

```bash
uv add moduna
pip install moduna
```

The framework extras are also available for teams that prefer explicit dependency declarations:

```bash
uv add "moduna[langchain]"
pip install "moduna[langchain]"

uv add "moduna[crewai]"
pip install "moduna[crewai]"
```

## Quick Start

```python
from moduna import Instruments, Moduna

moduna = Moduna()
moduna.init(
    {
        "app_name": "customer support",
        "framework": Instruments.Langchain,
        "api_key": "mod_...",
    }
)
```

Set a conversation ID using the Moduna-branded tracing namespace:

```python
from moduna.sdk.tracing import set_conversation_id

set_conversation_id("customer-123")
```

## Configuration

Moduna resolves the API key in this order:

1. `api_key` passed to `Moduna().init(...)`
2. exported `MODUNA_API_KEY`
3. `MODUNA_API_KEY=...` in a local `.env` file

The default API base URL is read from `config.toml`:

You can override it per application:

```python
from moduna import Instruments, Moduna

Moduna().init(
    {
        "app_name": "support-agent",
        "framework": Instruments.Langchain,
    }
)
```

## Frameworks

### LangChain

```python
from langchain_core.prompts import ChatPromptTemplate
from moduna import Instruments, Moduna
from moduna.sdk.tracing import set_conversation_id

Moduna().init({"app_name": "langchain-app", "framework": Instruments.Langchain})
set_conversation_id("conversation-1")

prompt = ChatPromptTemplate.from_messages([("human", "{question}")])
```

### CrewAI

```python
from moduna import Instruments, Moduna
from moduna.sdk.tracing import set_conversation_id

Moduna().init({"app_name": "crewai-app", "framework": Instruments.Crewai})
set_conversation_id("crew-run-1")
```

Moduna installs the Traceloop/OpenTelemetry CrewAI instrumentation. Install the CrewAI runtime separately in your
application when you build CrewAI agents.

## Development

Activate the local virtual environment before running checks:

```bash
source .venv/bin/activate
uv sync
uv run ruff format .
uv run ruff check .
uv run pytest
uv build
```

Use `uv run ruff format --check .` in CI when you want formatting verification without rewriting files.
