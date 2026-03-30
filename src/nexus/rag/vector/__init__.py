"""
Vector Storage Abstraction

Provides unified interface for vector storage backends.
Supports FAISS (development) and Milvus (production).
"""

from .client import VectorClient

__all__ = ["VectorClient"]
