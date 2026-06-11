"""Public API tests for the Moduna SDK."""

from moduna import AgentFramework, Moduna, ModunaConfiguration, __version__
from moduna.sdk.tracing import set_conversation_id


def test_public_exports_are_available() -> None:
    """The package should expose the documented public API."""
    assert Moduna is not None
    assert AgentFramework.Langchain == "langchain"
    assert ModunaConfiguration is not None
    assert __version__ == "0.1.9"
    assert callable(set_conversation_id)
