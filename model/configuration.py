from pydantic import BaseModel, ConfigDict, Field
from traceloop.sdk.instruments import Instruments


class ModunaConfiguration(BaseModel):
    """Configuration for Moduna tracing initialization."""

    model_config = ConfigDict(use_enum_values=False)

    base_url: str = Field(
        description="The base URL for the Moduna API",
        default="https://volex-506013021984.asia-south1.run.app",
    )
    api_key: str | None = Field(
        default=None,
        description="The API key for authenticating with the Moduna API",
    )
    framework: Instruments = Field(description="The agent framework to use")
    app_name: str = Field(description="The name of the application")
