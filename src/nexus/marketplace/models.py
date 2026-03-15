"""Shared data models for the Nexus marketplace system."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MarketplaceCategory(str, Enum):
    """Broad categories for marketplace listings."""

    # Provider categories
    LLM = "llm"
    EMBEDDING = "embedding"
    IMAGE = "image"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"
    CUSTOM = "custom"

    # Agent categories
    DATA_PROCESSING = "data_processing"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"
    AUTOMATION = "automation"
    RESEARCH = "research"
    MONITORING = "monitoring"


class ListingStatus(str, Enum):
    """Lifecycle status of a marketplace listing."""

    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    PUBLISHED = "published"
    SUSPENDED = "suspended"
    DEPRECATED = "deprecated"
    REMOVED = "removed"


@dataclass
class MarketplaceListing:
    """Base listing in either marketplace."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str = ""
    display_name: str = ""
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    author_url: str = ""
    category: MarketplaceCategory = MarketplaceCategory.CUSTOM
    tags: List[str] = field(default_factory=list)
    status: ListingStatus = ListingStatus.DRAFT
    license_feature: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "author_url": self.author_url,
            "category": self.category.value,
            "tags": list(self.tags),
            "status": self.status.value,
            "license_feature": self.license_feature,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }


@dataclass
class Review:
    """A user review for a marketplace listing."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    listing_id: str = ""
    reviewer: str = ""
    rating: int = 5  # 1-5
    title: str = ""
    body: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    helpful_count: int = 0
    verified: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.rating <= 5:
            raise ValueError(f"Rating must be 1-5, got {self.rating}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "listing_id": self.listing_id,
            "reviewer": self.reviewer,
            "rating": self.rating,
            "title": self.title,
            "body": self.body,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "helpful_count": self.helpful_count,
            "verified": self.verified,
        }


@dataclass
class UsageRecord:
    """A single usage event for a marketplace item."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    listing_id: str = ""
    user: str = ""
    action: str = "invoke"  # invoke | install | uninstall
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "listing_id": self.listing_id,
            "user": self.user,
            "action": self.action,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "tokens_used": self.tokens_used,
            "cost_usd": self.cost_usd,
            "success": self.success,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


@dataclass
class UsageSummary:
    """Aggregated usage statistics for a marketplace listing."""

    listing_id: str = ""
    total_invocations: int = 0
    total_installs: int = 0
    active_installs: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    success_rate: float = 1.0
    avg_duration_ms: float = 0.0
    unique_users: int = 0
    period_start: float = 0.0
    period_end: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "listing_id": self.listing_id,
            "total_invocations": self.total_invocations,
            "total_installs": self.total_installs,
            "active_installs": self.active_installs,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "success_rate": self.success_rate,
            "avg_duration_ms": self.avg_duration_ms,
            "unique_users": self.unique_users,
            "period_start": self.period_start,
            "period_end": self.period_end,
        }
