"""GraphQL-style DataLoader pattern for N+1 prevention."""

from nexus.core.database.dataloader.dataloader import DataLoader, DataLoaderRegistry

__all__ = ["DataLoader", "DataLoaderRegistry"]
