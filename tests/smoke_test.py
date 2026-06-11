"""Smoke test for built Moduna distributions."""

from importlib.metadata import version

from traceloop.sdk.instruments import Instruments as TraceloopInstruments

from moduna import Instruments, Moduna, ModunaConfiguration, __version__
from moduna.sdk.tracing import set_conversation_id


def main() -> None:
    """Verify the installed distribution exposes the documented API."""
    assert Instruments is TraceloopInstruments
    assert Moduna is not None
    assert ModunaConfiguration is not None
    assert __version__ == version("moduna")
    assert callable(set_conversation_id)


if __name__ == "__main__":
    main()
