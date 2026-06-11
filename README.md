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
        "framework": Instruments.LANGCHAIN,
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

The production Moduna endpoint is used by default. For local development, create
an ignored `config.toml` in the project root to switch to a local collector:

```toml
[sdk]
base_url = "http://localhost:4318"
```

You can also override the endpoint per application:

```python
from moduna import Instruments, Moduna

Moduna().init(
    {
        "app_name": "support-agent",
        "framework": Instruments.LANGCHAIN,
        "base_url": "http://localhost:4318",
    }
)
```

## Frameworks

### LangChain

```python
from langchain_core.prompts import ChatPromptTemplate
from moduna import Instruments, Moduna
from moduna.sdk.tracing import set_conversation_id

Moduna().init({"app_name": "langchain-app", "framework": Instruments.LANGCHAIN})
set_conversation_id("conversation-1")

prompt = ChatPromptTemplate.from_messages([("human", "{question}")])
```

### CrewAI

```python
from moduna import Instruments, Moduna
from moduna.sdk.tracing import set_conversation_id

Moduna().init({"app_name": "crewai-app", "framework": Instruments.CREWAI})
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

## Publishing

GitHub Actions validates pull requests and every push to `main`. A successful
`main` build publishes only when the version in `pyproject.toml` does not
already exist on PyPI.

Configure the PyPI token once:

1. Open the GitHub repository.
2. Go to **Settings → Secrets and variables → Actions**.
3. Add a repository secret named `PYPI_TOKEN`.
4. Paste the PyPI token value from your local `.env`.

Never commit `.env` or the token.

Prepare a release with uv:

```bash
source .venv/bin/activate
uv version --bump patch
git add pyproject.toml uv.lock
git commit -m "Release Moduna $(uv version --short)"
git push origin main
```

Changes that do not bump the version still run the full CI pipeline and skip
the PyPI upload successfully.
