---
name: Moduna Python SDK
description: Collects OTEL traces from agentic frameworks and llm apis.
---

# AGENTS.md

Moduna Python SDK. Light wrapper around Traceloop's OpenLLMetry SDK. Supports multiple frameworks.

---

## Setup and Run

- `source .venv/bin/active`: Use Virtual Enironment
- `uv sync`: Install all packages
- `python3 -m main`: Execute the main function

---

## Project Structure

- **model/**: All the types, constants and pydantic basemodel definition
- **services/**: All agents written for test and other SDK services
- **tools/**: All tools the testing agents can use. Used for testing tool and retrieval call traces
- **util/**: All utility functions that can be used across the package and are small sub routines or classes.

---

## SDK Ergonomics

Keep it simple and easy to integrate with any LLM or Agentic frameworks. Currently the SDK has one line of integration.

---

## Conversation ID

Conversation id can be set using `set_conversation_id()`. It is important to have a conversation id because the traces are analysed based on conversation.

## Publisher

PYPI publisher. 

## Repository

https://github.com/Moduna-AI/python-otel-sdk.git

This is the public repository.