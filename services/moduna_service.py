"""Moduna tracing service."""

from typing import Any

from traceloop.sdk import Traceloop
from traceloop.sdk.instruments import Instruments

from model.configuration import ModunaConfiguration
from services.configuration_service import ModunaConfigurationService


class Moduna:
    """Public SDK service for initializing Moduna tracing."""

    def __init__(
        self, configuration_service: ModunaConfigurationService | None = None
    ) -> None:
        """Initialize the SDK service."""
        self.configuration_service = (
            configuration_service or ModunaConfigurationService()
        )
        self.config: ModunaConfiguration | None = None
        self.client: Any | None = None

    def init(self, config: ModunaConfiguration | dict[str, Any]) -> None:
        """Initialize Traceloop tracing for the configured framework."""
        self.config = self.configuration_service.build(config)
        instrument = self.get_instrument(self.config.framework)
        self.client = Traceloop.init(
            app_name=self.config.app_name,
            api_endpoint=self.config.base_url,
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            instruments={instrument},
            endpoint_is_traceloop=False,
        )

    def get_instrument(self, framework: Instruments) -> Instruments:
        """Return the Traceloop instrument for a Moduna framework."""
        return framework
