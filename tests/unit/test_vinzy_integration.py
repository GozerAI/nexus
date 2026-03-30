"""Tests for Vinzy integration."""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from nexus.discovery.vinzy_integration import (
    ActivationResult,
    Entitlement,
    License,
    UsageResult,
    ValidationResult,
    VinzyConfig,
    VinzyIntegration,
)
from nexus.discovery.resource_discovery import (
    ResourceDiscovery,
    ResourceSource,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_discovery():
    rd = MagicMock(spec=ResourceDiscovery)
    rd.register_source = MagicMock()
    rd.register_resource = MagicMock(return_value=True)
    return rd


@pytest.fixture
def config():
    return VinzyConfig(
        server_url="http://test:8080",
        license_key="TST-ABCDE-FGHIJ-KLMNO-PQRST-UVWXY-12345-67890",
        api_key="test-api-key",
        product_id="TST",
    )


@pytest.fixture
def integration(mock_discovery, config):
    return VinzyIntegration(
        resource_discovery=mock_discovery,
        config=config,
    )


@pytest.fixture
def sample_license():
    return License(
        id="lic-001",
        key="TST-ABCDE-FGHIJ-KLMNO-PQRST-UVWXY-12345-67890",
        status="active",
        product_code="TST",
        customer_id="cust-001",
        tier="pro",
        machines_limit=5,
        machines_used=1,
        features=["feature_a", "feature_b"],
        entitlements={"feature_a": True},
    )


@pytest.fixture
def mock_client():
    """Create a mock Vinzy LicenseClient."""
    client = MagicMock()
    client.close = MagicMock()
    return client


# ── Config Tests ──────────────────────────────────────────────────────────────


class TestVinzyConfig:
    def test_defaults(self):
        config = VinzyConfig()
        assert config.server_url == "http://localhost:8080"
        assert config.license_key is None
        assert config.api_key is None
        assert config.product_id is None
        assert config.cache_ttl == 300
        assert config.timeout == 30.0

    def test_custom_values(self):
        config = VinzyConfig(server_url="http://custom:9090", license_key="KEY")
        assert config.server_url == "http://custom:9090"
        assert config.license_key == "KEY"


# ── Dataclass Tests ───────────────────────────────────────────────────────────


class TestDataclasses:
    def test_license(self, sample_license):
        assert sample_license.status == "active"
        assert sample_license.tier == "pro"
        assert len(sample_license.features) == 2

    def test_entitlement_available(self):
        ent = Entitlement(feature="pro_reports", enabled=True, limit=100, used=50, remaining=50)
        assert ent.is_available is True

    def test_entitlement_disabled(self):
        ent = Entitlement(feature="pro_reports", enabled=False)
        assert ent.is_available is False

    def test_entitlement_exhausted(self):
        ent = Entitlement(feature="api_calls", enabled=True, limit=100, used=100, remaining=0)
        assert ent.is_available is False

    def test_entitlement_unlimited(self):
        ent = Entitlement(feature="basic", enabled=True)
        assert ent.is_available is True

    def test_validation_result(self):
        result = ValidationResult(valid=True, code="OK", message="Valid license")
        assert result.valid is True
        assert result.features == []

    def test_activation_result(self):
        result = ActivationResult(success=True, machine_id="m-001", code="OK")
        assert result.success is True

    def test_usage_result(self):
        result = UsageResult(success=True, metric="api-calls", value_added=1.0, total_value=51.0)
        assert result.total_value == 51.0


# ── Init Tests ────────────────────────────────────────────────────────────────


class TestVinzyIntegrationInit:
    def test_init_with_config(self, integration, config):
        assert integration.config.server_url == "http://test:8080"
        assert integration.config.license_key is not None

    def test_init_without_discovery(self, config):
        integration = VinzyIntegration(config=config)
        assert integration.resource_discovery is None

    def test_init_from_env(self):
        with patch.dict("os.environ", {
            "VINZY_SERVER": "http://env:8080",
            "VINZY_LICENSE_KEY": "ENV-KEY",
            "VINZY_API_KEY": "env-api",
            "VINZY_PRODUCT_ID": "ENV",
        }):
            integration = VinzyIntegration()
            assert integration.config.server_url == "http://env:8080"
            assert integration.config.license_key == "ENV-KEY"


# ── Discovery Tests ───────────────────────────────────────────────────────────


class TestDiscover:
    @pytest.mark.asyncio
    async def test_discover_registers_service(self, integration, mock_discovery):
        count = await integration.discover()
        assert count == 1  # single vinzy-license-manager resource
        mock_discovery.register_resource.assert_called_once()

    @pytest.mark.asyncio
    async def test_discover_no_discovery_system(self, config):
        integration = VinzyIntegration(config=config)
        count = await integration.discover()
        assert count == 0

    @pytest.mark.asyncio
    async def test_discover_already_registered(self, integration, mock_discovery):
        mock_discovery.register_resource = MagicMock(return_value=False)
        count = await integration.discover()
        assert count == 0


# ── Fingerprint Tests ─────────────────────────────────────────────────────────


class TestFingerprint:
    def test_generates_fingerprint(self, integration):
        fp = integration._get_fingerprint()
        assert isinstance(fp, str)
        assert len(fp) == 32  # sha256 hex truncated to 32

    def test_fingerprint_cached(self, integration):
        fp1 = integration._get_fingerprint()
        fp2 = integration._get_fingerprint()
        assert fp1 == fp2


# ── Validation Tests ──────────────────────────────────────────────────────────


class TestValidation:
    @pytest.mark.asyncio
    async def test_validate_success(self, integration, mock_client):
        mock_license = MagicMock()
        mock_license.id = "lic-001"
        mock_license.key = "TST-KEY"
        mock_license.status = "active"
        mock_license.product_code = "TST"
        mock_license.customer_id = "cust-001"
        mock_license.tier = "pro"
        mock_license.machines_limit = 5
        mock_license.machines_used = 1
        mock_license.expires_at = None
        mock_license.features = ["feat_a"]
        mock_license.entitlements = {}

        mock_result = MagicMock()
        mock_result.valid = True
        mock_result.license = mock_license
        mock_result.code = "OK"
        mock_result.message = "Valid"
        mock_result.features = ["feat_a"]
        mock_result.entitlements = []

        mock_client.validate = MagicMock(return_value=mock_result)
        integration._client = mock_client

        result = await integration.validate()
        assert result.valid is True
        assert result.license.tier == "pro"
        assert integration._cached_license is not None

    @pytest.mark.asyncio
    async def test_validate_cached(self, integration, sample_license):
        integration._cached_license = sample_license
        integration._cache_timestamp = datetime.now(timezone.utc)

        result = await integration.validate()
        assert result.valid is True
        assert result.message == "Cached validation"

    @pytest.mark.asyncio
    async def test_validate_cache_expired(self, integration, sample_license, mock_client):
        integration._cached_license = sample_license
        integration._cache_timestamp = datetime.now(timezone.utc) - timedelta(seconds=600)

        mock_result = MagicMock()
        mock_result.valid = False
        mock_result.license = None
        mock_result.code = "EXPIRED"
        mock_result.message = "License expired"

        mock_client.validate = MagicMock(return_value=mock_result)
        integration._client = mock_client

        result = await integration.validate()
        assert result.valid is False

    @pytest.mark.asyncio
    async def test_validate_force_refresh(self, integration, sample_license, mock_client):
        integration._cached_license = sample_license
        integration._cache_timestamp = datetime.now(timezone.utc)

        mock_result = MagicMock()
        mock_result.valid = False
        mock_result.license = None
        mock_result.code = "SUSPENDED"
        mock_result.message = "Suspended"

        mock_client.validate = MagicMock(return_value=mock_result)
        integration._client = mock_client

        result = await integration.validate(force_refresh=True)
        assert result.valid is False

    @pytest.mark.asyncio
    async def test_validate_error(self, integration, mock_client):
        mock_client.validate = MagicMock(side_effect=Exception("Network error"))
        integration._client = mock_client

        result = await integration.validate()
        assert result.valid is False
        assert result.code == "ERROR"

    @pytest.mark.asyncio
    async def test_is_valid(self, integration, sample_license):
        integration._cached_license = sample_license
        integration._cache_timestamp = datetime.now(timezone.utc)

        assert await integration.is_valid() is True


# ── Entitlement Tests ─────────────────────────────────────────────────────────


class TestEntitlements:
    @pytest.mark.asyncio
    async def test_has_entitlement_cached(self, integration):
        integration._entitlement_cache["pro_reports"] = Entitlement(
            feature="pro_reports", enabled=True,
        )
        assert await integration.has_entitlement("pro_reports") is True

    @pytest.mark.asyncio
    async def test_has_entitlement_disabled(self, integration):
        integration._entitlement_cache["premium"] = Entitlement(
            feature="premium", enabled=False,
        )
        assert await integration.has_entitlement("premium") is False

    @pytest.mark.asyncio
    async def test_has_entitlement_missing(self, integration, sample_license):
        integration._cached_license = sample_license
        integration._cache_timestamp = datetime.now(timezone.utc)
        assert await integration.has_entitlement("nonexistent") is False

    @pytest.mark.asyncio
    async def test_get_entitlement(self, integration):
        ent = Entitlement(feature="api_calls", enabled=True, limit=1000, used=50, remaining=950)
        integration._entitlement_cache["api_calls"] = ent

        result = await integration.get_entitlement("api_calls")
        assert result.remaining == 950

    @pytest.mark.asyncio
    async def test_get_all_entitlements(self, integration, sample_license):
        integration._cached_license = sample_license
        integration._cache_timestamp = datetime.now(timezone.utc)
        integration._entitlement_cache["a"] = Entitlement(feature="a", enabled=True)
        integration._entitlement_cache["b"] = Entitlement(feature="b", enabled=True)

        result = await integration.get_all_entitlements()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_require_entitlement_decorator(self, integration):
        integration._entitlement_cache["pro"] = Entitlement(feature="pro", enabled=True)

        @integration.require_entitlement("pro")
        async def protected_func():
            return "ok"

        result = await protected_func()
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_require_entitlement_denied(self, integration, sample_license):
        integration._cached_license = sample_license
        integration._cache_timestamp = datetime.now(timezone.utc)

        @integration.require_entitlement("enterprise_only")
        async def protected_func():
            return "ok"

        with pytest.raises(PermissionError, match="enterprise_only"):
            await protected_func()


# ── Activation Tests ──────────────────────────────────────────────────────────


class TestActivation:
    @pytest.mark.asyncio
    async def test_activate_success(self, integration, mock_client):
        mock_license = MagicMock()
        mock_license.id = "lic-001"
        mock_license.key = "KEY"
        mock_license.status = "active"
        mock_license.product_code = "TST"
        mock_license.customer_id = "cust"
        mock_license.tier = "pro"
        mock_license.features = []
        mock_license.entitlements = {}

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.machine_id = "m-001"
        mock_result.license = mock_license
        mock_result.code = "OK"
        mock_result.message = "Activated"

        mock_client.activate = MagicMock(return_value=mock_result)
        integration._client = mock_client

        result = await integration.activate()
        assert result.success is True
        assert result.machine_id == "m-001"

    @pytest.mark.asyncio
    async def test_activate_failure(self, integration, mock_client):
        mock_client.activate = MagicMock(side_effect=Exception("Limit reached"))
        integration._client = mock_client

        result = await integration.activate()
        assert result.success is False
        assert result.code == "ERROR"

    @pytest.mark.asyncio
    async def test_deactivate_success(self, integration, mock_client):
        mock_client.deactivate = MagicMock(return_value=True)
        integration._client = mock_client
        integration._cached_license = MagicMock()

        result = await integration.deactivate()
        assert result is True
        assert integration._cached_license is None  # cache cleared

    @pytest.mark.asyncio
    async def test_deactivate_failure(self, integration, mock_client):
        mock_client.deactivate = MagicMock(side_effect=Exception("fail"))
        integration._client = mock_client

        result = await integration.deactivate()
        assert result is False


# ── Usage Tests ───────────────────────────────────────────────────────────────


class TestUsageTracking:
    @pytest.mark.asyncio
    async def test_record_usage_success(self, integration, mock_client):
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.value_added = 1.0
        mock_result.total_value = 101.0
        mock_result.limit = 1000.0
        mock_result.remaining = 899.0
        mock_result.code = "OK"

        mock_client.record_usage = MagicMock(return_value=mock_result)
        integration._client = mock_client

        result = await integration.record_usage("api-calls", value=1.0)
        assert result.success is True
        assert result.total_value == 101.0

    @pytest.mark.asyncio
    async def test_record_usage_failure(self, integration, mock_client):
        mock_client.record_usage = MagicMock(side_effect=Exception("fail"))
        integration._client = mock_client

        result = await integration.record_usage("api-calls")
        assert result.success is False
        assert result.code == "ERROR"

    @pytest.mark.asyncio
    async def test_heartbeat_success(self, integration, mock_client):
        mock_client.heartbeat = MagicMock(return_value=True)
        integration._client = mock_client

        result = await integration.heartbeat(version="2.0.0")
        assert result is True

    @pytest.mark.asyncio
    async def test_heartbeat_failure(self, integration, mock_client):
        mock_client.heartbeat = MagicMock(side_effect=Exception("fail"))
        integration._client = mock_client

        result = await integration.heartbeat()
        assert result is False


# ── Utility Tests ─────────────────────────────────────────────────────────────


class TestUtilities:
    def test_clear_cache(self, integration, sample_license):
        integration._cached_license = sample_license
        integration._cache_timestamp = datetime.now(timezone.utc)
        integration._entitlement_cache["feat"] = Entitlement(feature="feat", enabled=True)

        integration.clear_cache()
        assert integration._cached_license is None
        assert integration._cache_timestamp is None
        assert len(integration._entitlement_cache) == 0

    @pytest.mark.asyncio
    async def test_get_license_info_cached(self, integration, sample_license):
        integration._cached_license = sample_license
        integration._cache_timestamp = datetime.now(timezone.utc)

        info = await integration.get_license_info()
        assert info.tier == "pro"

    @pytest.mark.asyncio
    async def test_close(self, integration, mock_client):
        integration._client = mock_client
        integration._cached_license = MagicMock()

        await integration.close()
        mock_client.close.assert_called_once()
        assert integration._client is None
        assert integration._cached_license is None

    @pytest.mark.asyncio
    async def test_close_no_client(self, integration):
        await integration.close()

    @pytest.mark.asyncio
    async def test_health_check_unavailable(self, integration):
        with patch("aiohttp.ClientSession", side_effect=Exception("No connection")):
            result = await integration.health_check()
            assert result["status"] == "unavailable"
            assert result["service"] == "vinzy"
