"""
Embedding models for RAG system.

Supports multiple embedding providers:
- Local: sentence-transformers (no API calls)
- Ollama: Local models via Ollama
- OpenAI: text-embedding-3-small/large
- HuggingFace Inference API
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModel(ABC):
    """Base class for embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Embed a single text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts efficiently."""
        pass


class SentenceTransformerEmbedding(EmbeddingModel):
    """
    Local embeddings using sentence-transformers.

    Models:
    - all-MiniLM-L6-v2: 384 dim, fast, good quality (default)
    - all-mpnet-base-v2: 768 dim, better quality
    - bge-small-en-v1.5: 384 dim, excellent quality
    - bge-large-en-v1.5: 1024 dim, best quality
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dimension = None

    def _ensure_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info(f"Loaded {self.model_name} (dim={self._dimension})")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. Install with:\n"
                    "  pip install sentence-transformers"
                )

    @property
    def dimension(self) -> int:
        self._ensure_model()
        return self._dimension

    def embed(self, text: str) -> List[float]:
        self._ensure_model()
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self._ensure_model()
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings.tolist()


class OllamaEmbedding(EmbeddingModel):
    """
    Embeddings via Ollama.

    Models:
    - nomic-embed-text: 768 dim, good quality
    - mxbai-embed-large: 1024 dim, better quality
    - all-minilm: 384 dim, fast
    """

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self._dimension = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            # Get dimension by embedding a test string
            test_embedding = self.embed("test")
            self._dimension = len(test_embedding)
        return self._dimension

    def embed(self, text: str) -> List[float]:
        import requests

        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model_name, "prompt": text},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # Ollama doesn't have native batch, so we loop
        # Could parallelize with asyncio for better performance
        return [self.embed(text) for text in texts]


class OpenAIEmbedding(EmbeddingModel):
    """
    Embeddings via OpenAI API.

    Models:
    - text-embedding-3-small: 1536 dim, fast, cheap
    - text-embedding-3-large: 3072 dim, best quality
    - text-embedding-ada-002: 1536 dim, legacy
    """

    DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

        if model_name not in self.DIMENSIONS:
            logger.warning(f"Unknown model {model_name}, dimension lookup may fail")

    def _ensure_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "openai not installed. Install with:\n"
                    "  pip install openai"
                )

    @property
    def dimension(self) -> int:
        return self.DIMENSIONS.get(self.model_name, 1536)

    def embed(self, text: str) -> List[float]:
        self._ensure_client()
        response = self._client.embeddings.create(
            model=self.model_name,
            input=text,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self._ensure_client()
        response = self._client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]


class HuggingFaceEmbedding(EmbeddingModel):
    """
    Embeddings via HuggingFace Inference API.

    Uses the free inference API for embedding models.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("HUGGINGFACE_API_KEY")
        self._dimension = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            test_embedding = self.embed("test")
            self._dimension = len(test_embedding)
        return self._dimension

    def embed(self, text: str) -> List[float]:
        import requests

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(
            f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_name}",
            headers=headers,
            json={"inputs": text},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()

        # Handle nested arrays (some models return [[...]])
        if isinstance(result[0], list):
            return result[0]
        return result

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        import requests

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(
            f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_name}",
            headers=headers,
            json={"inputs": texts},
            timeout=60,
        )
        response.raise_for_status()
        results = response.json()

        # Handle nested arrays
        embeddings = []
        for result in results:
            if isinstance(result[0], list):
                embeddings.append(result[0])
            else:
                embeddings.append(result)
        return embeddings


def get_embedding_model(
    provider: str = "sentence-transformers",
    model_name: Optional[str] = None,
    **kwargs,
) -> EmbeddingModel:
    """
    Factory function to get an embedding model.

    Args:
        provider: One of "sentence-transformers", "ollama", "openai", "huggingface"
        model_name: Model name (uses sensible defaults if not specified)
        **kwargs: Additional arguments for the provider

    Returns:
        EmbeddingModel instance
    """
    providers = {
        "sentence-transformers": (SentenceTransformerEmbedding, "all-MiniLM-L6-v2"),
        "ollama": (OllamaEmbedding, "nomic-embed-text"),
        "openai": (OpenAIEmbedding, "text-embedding-3-small"),
        "huggingface": (HuggingFaceEmbedding, "sentence-transformers/all-MiniLM-L6-v2"),
    }

    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Choose from {list(providers.keys())}")

    model_class, default_model = providers[provider]
    model_name = model_name or default_model

    return model_class(model_name=model_name, **kwargs)
