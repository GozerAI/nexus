"""Unit tests for license gating enforcement."""

import pytest
from unittest.mock import patch, MagicMock

from nexus.licensing import LicenseGate, license_gate, _FEATURE_TIER_MAP, PRICING_URL


class TestLicenseGate:
    """Tests for LicenseGate core behavior."""

    def test_community_mode_when_no_key(self):
        gate = LicenseGate(license_key="")
        assert gate.is_community_mode is True

    def test_not_community_mode_when_key_set(self):
        gate = LicenseGate(license_key="test-key-123")
        assert gate.is_community_mode is False

    def test_check_feature_returns_false_in_community_mode(self):
        gate = LicenseGate(license_key="")
        assert gate.check_feature("nxs.reasoning.advanced") is False

    def test_gate_raises_permission_error_in_community_mode(self):
        gate = LicenseGate(license_key="")
        with pytest.raises(PermissionError, match="requires Pro license"):
            gate.gate("nxs.reasoning.advanced")

    def test_gate_raises_for_enterprise_feature(self):
        gate = LicenseGate(license_key="")
        with pytest.raises(PermissionError, match="requires Enterprise license"):
            gate.gate("nxs.discovery.intelligence")

    def test_gate_includes_pricing_url(self):
        gate = LicenseGate(license_key="")
        with pytest.raises(PermissionError, match=PRICING_URL):
            gate.gate("nxs.reasoning.advanced")

    def test_require_feature_decorator_blocks_in_community_mode(self):
        gate = LicenseGate(license_key="")

        @gate.require_feature("nxs.strategic.analysis")
        def protected_func():
            return "secret"

        with pytest.raises(PermissionError, match="Enterprise"):
            protected_func()

    def test_require_feature_decorator_allows_when_entitled(self):
        gate = LicenseGate(license_key="valid-key")
        gate._features_cache = ["nxs.strategic.analysis"]
        gate._cache_time = float("inf")

        @gate.require_feature("nxs.strategic.analysis")
        def protected_func():
            return "secret"

        assert protected_func() == "secret"

    def test_gate_allows_when_feature_entitled(self):
        gate = LicenseGate(license_key="valid-key")
        gate._features_cache = ["nxs.reasoning.advanced"]
        gate._cache_time = float("inf")
        # Should not raise
        gate.gate("nxs.reasoning.advanced")

    def test_gate_blocks_when_feature_not_in_entitlements(self):
        gate = LicenseGate(license_key="valid-key")
        gate._features_cache = ["nxs.reasoning.advanced"]
        gate._cache_time = float("inf")
        with pytest.raises(PermissionError, match="Enterprise"):
            gate.gate("nxs.discovery.intelligence")

    def test_vinzy_import_failure_falls_to_community(self):
        gate = LicenseGate(license_key="some-key")
        with patch.dict("sys.modules", {"vinzy_engine": None}):
            gate._client = None  # Reset cached client
            client = gate._get_client()
            assert client is None

    def test_refresh_features_returns_empty_without_client(self):
        gate = LicenseGate(license_key="some-key")
        features = gate._refresh_features()
        assert features == []

    def test_close_is_safe_when_no_client(self):
        gate = LicenseGate(license_key="")
        gate.close()  # Should not raise

    def test_all_feature_flags_have_tier_mapping(self):
        expected_flags = [
            "nxs.reasoning.advanced",
            "nxs.ensemble.multi_model",
            "nxs.discovery.intelligence",
            "nxs.strategic.analysis",
        ]
        for flag in expected_flags:
            assert flag in _FEATURE_TIER_MAP, f"Missing tier mapping for {flag}"

    def test_module_level_singleton_exists(self):
        """Module-level singleton should be a LicenseGate instance."""
        assert isinstance(license_gate, LicenseGate)


class TestLicenseGateEnforcement:
    """Tests verifying gates are actually wired into protected modules."""

    def test_reasoning_routes_import_license_gate(self):
        from nexus.api.routes import reasoning
        assert hasattr(reasoning, '_GATE')
        assert reasoning._GATE == "nxs.reasoning.advanced"

    def test_data_routes_import_license_gate(self):
        from nexus.api.routes import data
        assert hasattr(data, '_GATE')
        assert data._GATE == "nxs.discovery.intelligence"

    def test_unified_ensemble_gates_process(self):
        """Verify UnifiedEnsemble.process checks license."""
        import inspect
        from nexus.providers.ensemble.core import UnifiedEnsemble
        source = inspect.getsource(UnifiedEnsemble.process)
        assert "license_gate.gate" in source
        assert "nxs.ensemble.multi_model" in source

    def test_strategic_ensemble_gates_execute(self):
        """Verify StrategicEnsemble.execute_with_strategy checks license."""
        import inspect
        from nexus.core.strategic_ensemble import StrategicEnsemble
        source = inspect.getsource(StrategicEnsemble.execute_with_strategy)
        assert "license_gate.gate" in source
        assert "nxs.strategic.analysis" in source
