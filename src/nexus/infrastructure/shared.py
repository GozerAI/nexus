"""Shared infrastructure facade for Nexus.

This module defines the role Nexus should play when embedded into larger
products such as c-suite: shared AI infrastructure, not organizational
strategy ownership.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Optional, Sequence

from nexus.platform import NexusPlatform


@dataclass(frozen=True)
class SharedInfrastructureProfile:
    """Describes the architectural role Nexus owns."""

    name: str = "nexus"
    role: str = "shared_infrastructure"
    responsibilities: Sequence[str] = field(
        default_factory=lambda: (
            "multi_model_routing",
            "provider_registry",
            "memory_and_rag",
            "discovery",
            "observability",
            "execution_primitives",
            "interoperability_bridges",
        )
    )
    non_goals: Sequence[str] = field(
        default_factory=lambda: (
            "organizational_strategy",
            "executive_ownership",
            "autonomous_company_direction",
        )
    )
    control_plane_expectation: str = (
        "External products own strategic direction; Nexus provides shared AI services."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class SharedInfrastructureSnapshot:
    """Runtime view of the infrastructure role and service readiness."""

    profile: SharedInfrastructureProfile
    services: Dict[str, bool]

    @property
    def healthy(self) -> bool:
        """Return True when all initialized services are healthy."""
        return bool(self.services) and all(self.services.values())

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["healthy"] = self.healthy
        return payload


class NexusSharedInfrastructure:
    """Facade for using Nexus as shared AI infrastructure.

    The facade intentionally stops short of claiming strategic authority.
    It exposes the platform's reusable services so an external control plane
    can consume them.
    """

    def __init__(
        self,
        platform: Optional[NexusPlatform] = None,
        profile: Optional[SharedInfrastructureProfile] = None,
    ) -> None:
        self._platform = platform or NexusPlatform()
        self._profile = profile or SharedInfrastructureProfile()
        self._services: Optional[Dict[str, bool]] = None

    @property
    def platform(self) -> NexusPlatform:
        """Expose the underlying platform facade."""
        return self._platform

    @property
    def profile(self) -> SharedInfrastructureProfile:
        """Return the infrastructure profile."""
        return self._profile

    async def initialize(self, force_refresh: bool = False) -> Dict[str, bool]:
        """Initialize shared services."""
        if self._services is None or force_refresh:
            self._services = await self._platform.initialize()
        return dict(self._services)

    async def snapshot(self) -> SharedInfrastructureSnapshot:
        """Return the current service snapshot."""
        services = await self.initialize()
        return SharedInfrastructureSnapshot(profile=self._profile, services=services)

    def describe_role(self) -> Dict[str, Any]:
        """Return the static infrastructure contract."""
        return self._profile.to_dict()

    def get_service_matrix(self, services: Optional[Iterable[str]] = None) -> Dict[str, str]:
        """Describe what Nexus owns for each shared capability."""
        default_matrix = {
            "multi_model_routing": "owned",
            "provider_registry": "owned",
            "memory_and_rag": "owned",
            "discovery": "owned",
            "observability": "owned",
            "execution_primitives": "owned",
            "interoperability_bridges": "owned",
            "organizational_strategy": "external",
            "executive_ownership": "external",
            "autonomous_company_direction": "external",
        }
        if services is None:
            return default_matrix
        requested = set(services)
        return {name: status for name, status in default_matrix.items() if name in requested}
