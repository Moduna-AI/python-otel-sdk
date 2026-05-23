"""Shared pytest fixtures for Moduna OpenTelemetry tests."""

from __future__ import annotations

import pytest

from moduna_otel import ModunaOTEL


@pytest.fixture(autouse=True)
def reset_moduna_otel() -> None:
    """Reset singleton state so tests do not leak providers into each other."""
    ModunaOTEL._reset_for_tests()
    yield
    ModunaOTEL._reset_for_tests()
