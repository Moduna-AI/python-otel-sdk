"""TOML configuration helpers."""

import tomllib
from pathlib import Path
from typing import Any


class TomlConfigReader:
    """Read SDK defaults from a TOML config file."""

    def __init__(self, config_path: Path | str) -> None:
        """Initialize the reader with the target TOML path."""
        self.config_path = Path(config_path)

    def read(self) -> dict[str, Any]:
        """Read TOML configuration, returning an empty mapping when absent."""
        if not self.config_path.exists():
            return {}

        with self.config_path.open("rb") as config_file:
            return tomllib.load(config_file)
