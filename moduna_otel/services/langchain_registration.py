"""Best-effort global callback registration for LangChain."""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from typing import ClassVar, Protocol


class LangChainCallbackHandler(Protocol):
    """Small protocol for handlers that can be registered globally."""

    name: ClassVar[str]


_MODUNA_LANGCHAIN_HANDLER: ContextVar[LangChainCallbackHandler | None] = ContextVar(
    "moduna_otel_langchain_callback_handler",
    default=None,
)


def register_global_moduna_langchain_handler(
    handler: LangChainCallbackHandler,
    on_failure: Callable[[BaseException], None] | None = None,
) -> None:
    """Best-effort registration of a Moduna handler with Python LangChain globals."""
    try:
        from langchain_core.tracers.context import register_configure_hook

        register_configure_hook(_MODUNA_LANGCHAIN_HANDLER, inheritable=True)
        _MODUNA_LANGCHAIN_HANDLER.set(handler)
    except BaseException as exc:
        if on_failure is not None:
            on_failure(exc)
