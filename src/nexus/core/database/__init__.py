"""
Database persistence layer for TheNexus.

Provides SQLite-based persistence for:
- Users and API keys
- Cost tracking
- Usage analytics
- Strategy performance history
"""

from nexus.core.database.connection import DatabaseConnection, get_db, init_db
from nexus.core.database.models import (
    Base,
    UserModel,
    APIKeyModel,
    CostEntryModel,
    UsageEntryModel,
    StrategyPerformanceModel,
)

__all__ = [
    "DatabaseConnection",
    "get_db",
    "init_db",
    "Base",
    "UserModel",
    "APIKeyModel",
    "CostEntryModel",
    "UsageEntryModel",
    "StrategyPerformanceModel",
]
