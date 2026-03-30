"""
Database persistence layer for TheNexus.

Provides SQLite-based persistence for:
- Users and API keys
- Cost tracking
- Usage analytics
- Strategy performance history
"""

from nexus.core.database.connection import (
    DatabaseConnection,
    get_db,
    init_db,
    resolve_default_db_path,
)
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
    "resolve_default_db_path",
    "Base",
    "UserModel",
    "APIKeyModel",
    "CostEntryModel",
    "UsageEntryModel",
    "StrategyPerformanceModel",
]
