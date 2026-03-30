"""Tests for Zuultimate integration."""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from nexus.discovery.zuultimate_integration import (
    AuthorizationResult,
    IdentityToken,
    TrustScore,
    VaultToken,
    ZuultimateConfig,
    ZuultimateIntegration,
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
    return ZuultimateConfig(
        base_url="http://test:8000",
        api_key="test-key",
        access_token="test-token",
    )


@pytest.fixture
def integration(mock_discovery, config):
    return ZuultimateIntegration(
        resource_discovery=mock_discovery,
        config=config,
    )


@pytest.fixture
def mock_client():
    """Create a mock ZuultimateClient."""
    client = AsyncMock()
    client.identity = AsyncMock()
    client.vault = AsyncMock()
    client.access = AsyncMock()
    client.close = AsyncMock()
    return client


# ── Config Tests ──────────────────────────────────────────────────────────────


class TestZuultimateConfig:
    def test_defaults(self):
        config = ZuultimateConfig()
        assert config.base_url == "http://localhost:8000"
        assert config.api_key is None
        assert config.access_token is None
        assert config.timeout == 30.0
        assert config.cache_ttl == 300

    def test_custom_values(self):
        config = ZuultimateConfig(base_url="http://custom:9999", api_key="key123")
        assert config.base_url == "http://custom:9999"
        assert config.api_key == "key123"


# ── Dataclass Tests ───────────────────────────────────────────────────────────


class TestDataclasses:
    def test_identity_token(self):
        now = datetime.now(timezone.utc)
        token = IdentityToken(
            token_id="tok-123",
            created_at=now,
            expires_at=now + timedelta(hours=1),
        )
        assert token.is_valid is True
        assert token.token_id == "tok-123"

    def test_trust_score(self):
        score = TrustScore(score=0.85, risk_level="low", factors={"login_count": 100})
        assert score.score == 0.85
        assert score.risk_level == "low"
        assert score.factors["login_count"] == 100

    def test_vault_token(self):
        vt = VaultToken(
            token="vtk-abc",
            namespace="pii",
            created_at=datetime.now(timezone.utc),
        )
        assert vt.token == "vtk-abc"
        assert vt.namespace == "pii"

    def test_authorization_result(self):
        result = AuthorizationResult(allowed=True, permission="PII_REVEAL")
        assert result.allowed is True
        assert result.reason is None


# ── Integration Init Tests ────────────────────────────────────────────────────


class TestZuultimateIntegrationInit:
    def test_init_with_config(self, integration, config):
        assert integration.config.base_url == "http://test:8000"
        assert integration.config.api_key == "test-key"

    def test_init_registers_with_discovery(self, mock_discovery, config):
        ZuultimateIntegration(resource_discovery=mock_discovery, config=config)
        mock_discovery.register_source.assert_called_once_with(
            ResourceSource.ZUULTIMATE, pytest.approx(mock_discovery.register_source.call_args[0][1], abs=1),
        )

    def test_init_without_discovery(self, config):
        integration = ZuultimateIntegration(config=config)
        assert integration.resource_discovery is None

    def test_init_from_env(self):
        with patch.dict("os.environ", {
            "ZUULTIMATE_URL": "http://env:8000",
            "ZUULTIMATE_API_KEY": "env-key",
            "ZUULTIMATE_ACCESS_TOKEN": "env-token",
        }):
            integration = ZuultimateIntegration()
            assert integration.config.base_url == "http://env:8000"
            assert integration.config.api_key == "env-key"


# ── Discovery Tests ───────────────────────────────────────────────────────────


class TestDiscover:
    @pytest.mark.asyncio
    async def test_discover_registers_services(self, integration, mock_discovery):
        count = await integration.discover()
        assert count == 3  # identity, vault, access
        assert mock_discovery.register_resource.call_count == 3

    @pytest.mark.asyncio
    async def test_discover_no_discovery_system(self, config):
        integration = ZuultimateIntegration(config=config)
        count = await integration.discover()
        assert count == 0

    @pytest.mark.asyncio
    async def test_discover_already_registered(self, integration, mock_discovery):
        mock_discovery.register_resource = MagicMock(return_value=False)
        count = await integration.discover()
        assert count == 0


# ── Identity Tests ────────────────────────────────────────────────────────────


class TestIdentityOperations:
    @pytest.mark.asyncio
    async def test_authenticate_success(self, integration, mock_client):
        mock_client.identity.login.return_value = {
            "access_token": "at-123",
            "refresh_token": "rt-456",
        }
        integration._client = mock_client

        result = await integration.authenticate("user", "pass")
        assert result["access_token"] == "at-123"
        mock_client.identity.login.assert_called_once_with(
            username="user", password="pass", mfa_code=None,
        )

    @pytest.mark.asyncio
    async def test_authenticate_with_mfa(self, integration, mock_client):
        mock_client.identity.login.return_value = {"access_token": "at"}
        integration._client = mock_client

        await integration.authenticate("user", "pass", mfa_code="123456")
        mock_client.identity.login.assert_called_once_with(
            username="user", password="pass", mfa_code="123456",
        )

    @pytest.mark.asyncio
    async def test_authenticate_failure(self, integration, mock_client):
        mock_client.identity.login.side_effect = Exception("Invalid credentials")
        integration._client = mock_client

        with pytest.raises(Exception, match="Invalid credentials"):
            await integration.authenticate("user", "wrong")

    @pytest.mark.asyncio
    async def test_validate_token_success(self, integration, mock_client):
        now = datetime.now(timezone.utc)
        mock_client.identity.validate_token.return_value = {
            "token_id": "tok-abc",
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(hours=1)).isoformat(),
            "valid": True,
        }
        integration._client = mock_client

        result = await integration.validate_token("my-token")
        assert result.is_valid is True
        assert result.token_id == "tok-abc"
        # Should be cached
        assert "my-token" in integration._identity_cache

    @pytest.mark.asyncio
    async def test_validate_token_cached(self, integration):
        now = datetime.now(timezone.utc)
        cached = IdentityToken(
            token_id="cached",
            created_at=now,
            expires_at=now + timedelta(hours=1),
            is_valid=True,
        )
        integration._identity_cache["cached-token"] = cached

        result = await integration.validate_token("cached-token")
        assert result.token_id == "cached"

    @pytest.mark.asyncio
    async def test_validate_token_failure(self, integration, mock_client):
        mock_client.identity.validate_token.side_effect = Exception("Network error")
        integration._client = mock_client

        result = await integration.validate_token("bad-token")
        assert result.is_valid is False
        assert result.token_id == "invalid"

    @pytest.mark.asyncio
    async def test_get_trust_score_success(self, integration, mock_client):
        mock_client.identity.get_trust_score.return_value = {
            "score": 0.9,
            "risk_level": "low",
            "factors": {"login_history": "clean"},
        }
        integration._client = mock_client

        result = await integration.get_trust_score("user-123")
        assert result.score == 0.9
        assert result.risk_level == "low"

    @pytest.mark.asyncio
    async def test_get_trust_score_cached(self, integration, mock_client):
        mock_client.identity.get_trust_score.return_value = {
            "score": 0.9,
            "risk_level": "low",
        }
        integration._client = mock_client

        # First call
        await integration.get_trust_score("user-123")
        # Second call should use cache
        await integration.get_trust_score("user-123")
        assert mock_client.identity.get_trust_score.call_count == 1

    @pytest.mark.asyncio
    async def test_get_trust_score_failure(self, integration, mock_client):
        mock_client.identity.get_trust_score.side_effect = Exception("fail")
        integration._client = mock_client

        result = await integration.get_trust_score("user-123")
        assert result.score == 0.0
        assert result.risk_level == "high"

    @pytest.mark.asyncio
    async def test_logout_success(self, integration, mock_client):
        integration._client = mock_client
        result = await integration.logout("session-123")
        assert result is True
        mock_client.identity.logout.assert_called_once_with("session-123")

    @pytest.mark.asyncio
    async def test_logout_failure(self, integration, mock_client):
        mock_client.identity.logout.side_effect = Exception("fail")
        integration._client = mock_client
        result = await integration.logout()
        assert result is False


# ── Vault Tests ───────────────────────────────────────────────────────────────


class TestVaultOperations:
    @pytest.mark.asyncio
    async def test_encrypt_field(self, integration, mock_client):
        mock_client.vault.encrypt.return_value = {"reference_id": "ref-123"}
        integration._client = mock_client

        result = await integration.encrypt_field("owner-1", "email", "test@example.com")
        assert result == "ref-123"

    @pytest.mark.asyncio
    async def test_encrypt_field_failure(self, integration, mock_client):
        mock_client.vault.encrypt.side_effect = Exception("Vault error")
        integration._client = mock_client

        with pytest.raises(Exception, match="Vault error"):
            await integration.encrypt_field("owner-1", "email", "test@example.com")

    @pytest.mark.asyncio
    async def test_decrypt_field(self, integration, mock_client):
        mock_client.vault.decrypt.return_value = {"value": "decrypted@example.com"}
        integration._client = mock_client

        result = await integration.decrypt_field("owner-1", "email")
        assert result == "decrypted@example.com"

    @pytest.mark.asyncio
    async def test_tokenize(self, integration, mock_client):
        mock_client.vault.tokenize.return_value = {"token": "vtk-abc"}
        integration._client = mock_client

        result = await integration.tokenize("sensitive-value", namespace="pii")
        assert result.token == "vtk-abc"
        assert result.namespace == "pii"

    @pytest.mark.asyncio
    async def test_detokenize(self, integration, mock_client):
        mock_client.vault.detokenize.return_value = {"value": "original"}
        integration._client = mock_client

        result = await integration.detokenize("vtk-abc")
        assert result == "original"


# ── Authorization Tests ───────────────────────────────────────────────────────


class TestAuthorizationOperations:
    @pytest.mark.asyncio
    async def test_check_permission_allowed(self, integration, mock_client):
        mock_client.access.check_permission.return_value = {
            "allowed": True,
            "reason": "Role match",
        }
        integration._client = mock_client

        result = await integration.check_permission("PII_REVEAL")
        assert result.allowed is True
        assert result.permission == "PII_REVEAL"

    @pytest.mark.asyncio
    async def test_check_permission_denied(self, integration, mock_client):
        mock_client.access.check_permission.return_value = {
            "allowed": False,
            "reason": "Insufficient role",
        }
        integration._client = mock_client

        result = await integration.check_permission("ADMIN_DELETE")
        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_check_permission_error(self, integration, mock_client):
        mock_client.access.check_permission.side_effect = Exception("fail")
        integration._client = mock_client

        result = await integration.check_permission("ANYTHING")
        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_authorize_success(self, integration, mock_client):
        mock_client.access.check_permission.return_value = {"allowed": True}
        integration._client = mock_client

        result = await integration.authorize("PII_REVEAL")
        assert result is True

    @pytest.mark.asyncio
    async def test_authorize_denied_raises(self, integration, mock_client):
        mock_client.access.check_permission.return_value = {
            "allowed": False,
            "reason": "No access",
        }
        integration._client = mock_client

        with pytest.raises(PermissionError, match="Access denied"):
            await integration.authorize("PII_REVEAL")

    @pytest.mark.asyncio
    async def test_authorize_denied_no_raise(self, integration, mock_client):
        mock_client.access.check_permission.return_value = {"allowed": False}
        integration._client = mock_client

        result = await integration.authorize("PII_REVEAL", raise_on_deny=False)
        assert result is False

    @pytest.mark.asyncio
    async def test_get_user_permissions(self, integration, mock_client):
        mock_client.access.get_permissions.return_value = {
            "permissions": ["PII_REVEAL", "VAULT_READ"],
        }
        integration._client = mock_client

        perms = await integration.get_user_permissions()
        assert "PII_REVEAL" in perms
        assert len(perms) == 2

    @pytest.mark.asyncio
    async def test_get_user_permissions_error(self, integration, mock_client):
        mock_client.access.get_permissions.side_effect = Exception("fail")
        integration._client = mock_client

        perms = await integration.get_user_permissions()
        assert perms == []

    @pytest.mark.asyncio
    async def test_get_user_roles(self, integration, mock_client):
        mock_client.access.get_roles.return_value = {
            "roles": [{"name": "admin", "level": 10}],
        }
        integration._client = mock_client

        roles = await integration.get_user_roles()
        assert len(roles) == 1
        assert roles[0]["name"] == "admin"


# ── Utility Tests ─────────────────────────────────────────────────────────────


class TestUtilities:
    def test_clear_cache(self, integration):
        now = datetime.now(timezone.utc)
        integration._identity_cache["tok"] = IdentityToken(
            token_id="t", created_at=now, expires_at=now,
        )
        integration._trust_cache["key"] = TrustScore(score=0.5, risk_level="med")

        integration.clear_cache()
        assert len(integration._identity_cache) == 0
        assert len(integration._trust_cache) == 0

    @pytest.mark.asyncio
    async def test_close(self, integration, mock_client):
        integration._client = mock_client
        integration._identity_cache["tok"] = MagicMock()

        await integration.close()
        mock_client.close.assert_called_once()
        assert integration._client is None
        assert len(integration._identity_cache) == 0

    @pytest.mark.asyncio
    async def test_close_no_client(self, integration):
        # Should not raise
        await integration.close()

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, integration):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "ok"})

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(),
        ))

        with patch("aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cls.return_value.__aexit__ = AsyncMock()

            result = await integration.health_check()
            assert result["status"] == "healthy"
            assert result["service"] == "zuultimate"

    @pytest.mark.asyncio
    async def test_health_check_unavailable(self, integration):
        with patch("aiohttp.ClientSession", side_effect=Exception("No connection")):
            result = await integration.health_check()
            assert result["status"] == "unavailable"
