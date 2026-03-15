"""Regression tests for cog_eng configuration logging."""

from types import SimpleNamespace
from unittest.mock import patch

from nexus.cog_eng.config import Config


def test_validate_uses_info_for_default_dev_warnings():
    config = Config.__new__(Config)
    config.environment = "development"
    config.llm = SimpleNamespace(
        openai_api_key=None,
        anthropic_api_key=None,
        google_api_key=None,
    )
    config.security = SimpleNamespace(signing_key="changeme-secret")

    with patch("nexus.cog_eng.config.logger.info") as mock_info:
        with patch("nexus.cog_eng.config.logger.warning") as mock_warning:
            Config._validate(config)

    assert mock_warning.call_count == 0
    assert mock_info.call_count == 2


def test_validate_uses_warning_in_production():
    config = Config.__new__(Config)
    config.environment = "production"
    config.llm = SimpleNamespace(
        openai_api_key=None,
        anthropic_api_key=None,
        google_api_key=None,
    )
    config.security = SimpleNamespace(signing_key="changeme-secret")

    with patch("nexus.cog_eng.config.logger.info") as mock_info:
        with patch("nexus.cog_eng.config.logger.warning") as mock_warning:
            Config._validate(config)

    assert mock_info.call_count == 0
    assert mock_warning.call_count == 2
