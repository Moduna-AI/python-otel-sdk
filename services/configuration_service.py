"""Configuration assembly for the Moduna SDK."""

import os
from pathlib import Path

from model.configuration import ModunaConfiguration
from util.config import TomlConfigReader
from util.env import EnvFileReader

MODUNA_API_KEY_ENV = "MODUNA_API_KEY"


class ModunaConfigurationService:
    """Build complete Moduna configuration from defaults, env, and user input."""

    def __init__(
        self,
        config_path: Path | str | None = None,
        env_path: Path | str = ".env",
    ) -> None:
        """Initialize the configuration service."""
        self.config_path = (
            Path(config_path)
            if config_path is not None
            else self.default_config_path()
        )
        self.env_reader = EnvFileReader(env_path)

    from typing import Any

    def build(
        self, config: ModunaConfiguration | dict[str, Any]
    ) -> ModunaConfiguration:
        """Build a complete SDK configuration."""
        # 1. Safely handle the incoming config
        user_config = (
            config.model_dump()
            if isinstance(config, ModunaConfiguration)
            else dict(config)
        )

        # 2. Load defaults from the TOML file
        defaults = TomlConfigReader(self.config_path).read()
        sdk_defaults = (
            defaults.get("sdk", {}) if isinstance(defaults, dict) else {}
        )
        print(
            f"Loaded Moduna SDK defaults from {self.config_path}: {sdk_defaults}"
        )

        # 3. Resolve the base_url
        # Fallback order: User Config -> TOML Defaults -> Pydantic Model Default
        base_url = user_config.get("base_url") or sdk_defaults.get("base_url")

        if base_url:
            resolved_config = {"base_url": base_url, **user_config}
        else:
            # If neither user nor TOML provided it, let Pydantic use its own default
            resolved_config = {**user_config}

        # 4. Resolve the API key
        resolved_config["api_key"] = self.resolve_api_key(
            user_config.get("api_key")
        )

        return ModunaConfiguration(**resolved_config)

    def resolve_api_key(self, explicit_api_key: str | None) -> str:
        """Resolve the API key from explicit config, environment, then dotenv."""
        api_key = (
            explicit_api_key
            or os.environ.get(MODUNA_API_KEY_ENV)
            or self.env_reader.get(MODUNA_API_KEY_ENV)
        )
        if not api_key:
            raise ValueError(
                "Moduna API key is required. Set api_key or MODUNA_API_KEY."
            )

        return api_key

    def default_config_path(self) -> Path:
        """Return the default config.toml path bundled with the package."""
        return Path(__file__).resolve().parents[1] / "config.toml"
