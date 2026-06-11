"""Configuration tests for the Moduna SDK."""

from pathlib import Path

import pytest

from model.configuration import Instruments
from services.configuration_service import (
    MODUNA_API_KEY_ENV,
    ModunaConfigurationService,
)


def write_config(tmp_path: Path) -> Path:
    """Write a test SDK config file."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[sdk]\nbase_url = "http://localhost:4318"\n', encoding="utf-8"
    )
    return config_path


def test_explicit_api_key_wins_over_env_and_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Explicit API keys should have the highest precedence."""
    env_path = tmp_path / ".env"
    env_path.write_text("MODUNA_API_KEY=mod_dotenv\n", encoding="utf-8")
    monkeypatch.setenv(MODUNA_API_KEY_ENV, "mod_env")

    config = ModunaConfigurationService(
        config_path=write_config(tmp_path), env_path=env_path
    ).build(
        {
            "app_name": "support",
            "framework": Instruments.LANGCHAIN,
            "api_key": "mod_explicit",
        }
    )

    assert config.api_key == "mod_explicit"


def test_environment_api_key_is_used_when_config_omits_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The exported API key should be used when config omits api_key."""
    monkeypatch.setenv(MODUNA_API_KEY_ENV, "mod_env")

    config = ModunaConfigurationService(
        config_path=write_config(tmp_path), env_path=tmp_path / ".env"
    ).build(
        {
            "app_name": "support",
            "framework": Instruments.LANGCHAIN,
        }
    )

    assert config.api_key == "mod_env"


def test_dotenv_api_key_is_used_when_environment_omits_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The local dotenv value should be used after explicit and exported keys."""
    monkeypatch.delenv(MODUNA_API_KEY_ENV, raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("MODUNA_API_KEY=mod_dotenv\n", encoding="utf-8")

    config = ModunaConfigurationService(
        config_path=write_config(tmp_path), env_path=env_path
    ).build(
        {
            "app_name": "support",
            "framework": Instruments.LANGCHAIN,
        }
    )

    assert config.api_key == "mod_dotenv"


def test_missing_api_key_raises_value_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing API key should fail before tracing is initialized."""
    monkeypatch.delenv(MODUNA_API_KEY_ENV, raising=False)

    with pytest.raises(ValueError, match="Moduna API key is required"):
        ModunaConfigurationService(
            config_path=write_config(tmp_path), env_path=tmp_path / ".env"
        ).build(
            {
                "app_name": "support",
                "framework": Instruments.LANGCHAIN,
            }
        )


def test_public_framework_names_are_accepted(tmp_path: Path) -> None:
    """The README framework names should parse as enum values."""
    config = ModunaConfigurationService(
        config_path=write_config(tmp_path)
    ).build(
        {
            "app_name": "support",
            "framework": "langchain",
            "api_key": "mod_test",
        }
    )

    assert config.framework == Instruments.LANGCHAIN


def test_local_config_overrides_production_base_url(tmp_path: Path) -> None:
    """A local config file should override the production collector URL."""
    config = ModunaConfigurationService(
        config_path=write_config(tmp_path)
    ).build(
        {
            "app_name": "support",
            "framework": Instruments.LANGCHAIN,
            "api_key": "mod_test",
        }
    )

    assert config.base_url == "http://localhost:4318"


def test_missing_local_config_uses_production_base_url(tmp_path: Path) -> None:
    """The model default should be used when local config is absent."""
    config = ModunaConfigurationService(
        config_path=tmp_path / "missing.toml"
    ).build(
        {
            "app_name": "support",
            "framework": Instruments.LANGCHAIN,
            "api_key": "mod_test",
        }
    )

    assert config.base_url == ("https://volex-506013021984.asia-south1.run.app")
