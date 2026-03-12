"""Custom exceptions for Nexus Intelligence Platform."""


class NexusError(Exception):
    """Base exception."""
    pass


class StorageError(NexusError):
    """Storage operation failed."""
    pass


class EmbeddingError(NexusError):
    """Embedding generation failed."""
    pass


class NotFoundError(NexusError):
    """Resource not found."""
    pass


class DuplicateError(NexusError):
    """Duplicate detected."""
    pass


class VerificationError(NexusError):
    """Truth verification failed."""
    pass
