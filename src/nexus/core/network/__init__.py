"""Network optimization modules for Nexus."""

from nexus.core.network.request_batching import RequestBatcher, BatchedRequest
from nexus.core.network.compression import CompressionMiddleware, CompressionAlgorithm
from nexus.core.network.provider_queue import ProviderRequestQueue, QueuedProviderRequest

__all__ = [
    "RequestBatcher",
    "BatchedRequest",
    "CompressionMiddleware",
    "CompressionAlgorithm",
    "ProviderRequestQueue",
    "QueuedProviderRequest",
]
