"""Tests for model providers."""

import os
import sys
import pytest
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from nexus.core.models.base import ModelConfig, ModelProvider, ModelResponse
from nexus.core.models.stub_provider import StubProvider
from nexus.core.models.model_factory import ModelFactory


class TestModelBase:
    """Tests for base model classes."""

    def test_model_config_creation(self):
        """Test ModelConfig creation."""
        config = ModelConfig(
            name="test-model",
            provider=ModelProvider.STUB,
            weight=0.7,
            temperature=0.8,
            max_tokens=500,
        )
        
        assert config.name == "test-model"
        assert config.provider == ModelProvider.STUB
        assert config.weight == 0.7
        assert config.temperature == 0.8
        assert config.max_tokens == 500

    def test_model_response_success(self):
        """Test successful ModelResponse."""
        response = ModelResponse(
            content="Test response",
            model_name="test-model",
            provider="stub",
            tokens_used=10,
            latency_ms=100.0,
        )
        
        assert response.success
        assert response.content == "Test response"
        assert response.error is None

    def test_model_response_error(self):
        """Test ModelResponse with error."""
        response = ModelResponse(
            content="",
            model_name="test-model",
            provider="stub",
            error="Test error",
        )
        
        assert not response.success
        assert response.error == "Test error"


class TestStubProvider:
    """Tests for StubProvider."""

    def test_stub_provider_init(self):
        """Test StubProvider initialization."""
        config = ModelConfig(
            name="test-stub",
            provider=ModelProvider.STUB,
        )
        
        provider = StubProvider(config)
        assert provider.name == "test-stub"
        assert provider.provider == ModelProvider.STUB

    def test_stub_provider_validate(self):
        """Test StubProvider validation."""
        config = ModelConfig(
            name="test-stub",
            provider=ModelProvider.STUB,
        )
        
        provider = StubProvider(config)
        assert provider.validate_config()

    @pytest.mark.asyncio
    async def test_stub_provider_generate(self):
        """Test StubProvider generate method."""
        config = ModelConfig(
            name="test-stub",
            provider=ModelProvider.STUB,
        )
        
        provider = StubProvider(config)
        response = await provider.generate("Test prompt")
        
        assert response.success
        assert "Test prompt" in response.content
        assert response.model_name == "test-stub"
        assert response.tokens_used > 0
        assert response.latency_ms > 0
        assert response.cost == 0.0

    def test_stub_provider_generate_sync(self):
        """Test StubProvider synchronous generation."""
        config = ModelConfig(
            name="test-stub",
            provider=ModelProvider.STUB,
        )
        
        provider = StubProvider(config)
        response = provider.generate_sync("Test prompt")
        
        assert response.success
        assert "Test prompt" in response.content


class TestModelFactory:
    """Tests for ModelFactory."""

    def test_create_stub_model(self):
        """Test creating a stub model."""
        config = ModelConfig(
            name="test-stub",
            provider=ModelProvider.STUB,
        )
        
        model = ModelFactory.create_model(config)
        assert isinstance(model, StubProvider)
        assert model.name == "test-stub"

    def test_create_model_invalid_provider(self):
        """Test creating model with invalid provider."""
        # This should work with stub as fallback
        config = ModelConfig(
            name="test-model",
            provider=ModelProvider.STUB,
        )
        
        model = ModelFactory.create_model(config)
        assert model is not None

    def test_list_providers(self):
        """Test listing available providers."""
        providers = ModelFactory.list_providers()
        assert "stub" in providers
        assert "openai" in providers
        assert "anthropic" in providers
