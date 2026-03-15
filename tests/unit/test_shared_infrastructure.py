"""Tests for the shared-infrastructure Nexus contract."""

import asyncio

from nexus.infrastructure import NexusSharedInfrastructure
from nexus.service import legacy_coo_enabled_from_env


class StubPlatform:
    """Simple async platform stub for infrastructure tests."""

    def __init__(self, status):
        self.status = status
        self.calls = 0

    async def initialize(self):
        self.calls += 1
        return dict(self.status)


def test_profile_positions_nexus_as_shared_infrastructure():
    infra = NexusSharedInfrastructure(platform=StubPlatform({"ensemble": True}))

    profile = infra.describe_role()

    assert profile["role"] == "shared_infrastructure"
    assert "multi_model_routing" in profile["responsibilities"]
    assert "organizational_strategy" in profile["non_goals"]
    assert "External products own strategic direction" in profile["control_plane_expectation"]


def test_snapshot_caches_platform_initialization():
    stub = StubPlatform({"ensemble": True, "observatory": True})
    infra = NexusSharedInfrastructure(platform=stub)

    first = asyncio.run(infra.snapshot())
    second = asyncio.run(infra.snapshot())

    assert first.healthy is True
    assert second.services == {"ensemble": True, "observatory": True}
    assert stub.calls == 1


def test_service_matrix_marks_strategy_as_external():
    infra = NexusSharedInfrastructure(platform=StubPlatform({"ensemble": True}))

    matrix = infra.get_service_matrix()

    assert matrix["multi_model_routing"] == "owned"
    assert matrix["organizational_strategy"] == "external"
    assert matrix["autonomous_company_direction"] == "external"


def test_legacy_coo_toggle_defaults_off():
    assert legacy_coo_enabled_from_env({}) is False
    assert legacy_coo_enabled_from_env({"NEXUS_ENABLE_LEGACY_COO": "false"}) is False
    assert legacy_coo_enabled_from_env({"NEXUS_ENABLE_LEGACY_COO": "true"}) is True
