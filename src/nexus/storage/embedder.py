"""Local embedding generation - supports both built-in and Ollama backends."""

import asyncio
import hashlib
import subprocess
import sys
import os
import time
from typing import List, Optional, Literal
from dataclasses import dataclass, field
from pathlib import Path
import httpx

from nexus.core.exceptions import EmbeddingError


@dataclass
class EmbeddingConfig:
    # Backend selection: "auto", "builtin", "ollama"
    backend: Literal["auto", "builtin", "ollama"] = "auto"

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "nomic-embed-text"
    ollama_fallback_model: str = "mxbai-embed-large"
    auto_start_ollama: bool = True

    # Built-in settings (sentence-transformers)
    builtin_model: str = "all-MiniLM-L6-v2"  # Fast, good quality, ~80MB

    # Common settings
    batch_size: int = 32
    timeout_seconds: float = 30.0
    cache_enabled: bool = True

    # Legacy aliases
    @property
    def primary_model(self):
        return self.ollama_model

    @property
    def fallback_model(self):
        return self.ollama_fallback_model


class OllamaManager:
    """Manages Ollama process lifecycle."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self._process: Optional[subprocess.Popen] = None

    async def is_running(self) -> bool:
        """Check if Ollama is running and responsive."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False

    async def start(self) -> bool:
        """Start Ollama if not running. Returns True if started successfully."""
        if await self.is_running():
            return True

        # Find ollama executable
        ollama_cmd = self._find_ollama()
        if not ollama_cmd:
            return False

        try:
            # Start ollama serve in background
            if sys.platform == "win32":
                # Windows: use CREATE_NO_WINDOW to hide console
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                self._process = subprocess.Popen(
                    [ollama_cmd, "serve"],
                    startupinfo=startupinfo,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:
                # Unix: standard background process
                self._process = subprocess.Popen(
                    [ollama_cmd, "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

            # Wait for it to become responsive
            for _ in range(30):  # Wait up to 15 seconds
                await asyncio.sleep(0.5)
                if await self.is_running():
                    return True

            return False

        except Exception as e:
            print(f"Failed to start Ollama: {e}")
            return False

    def _find_ollama(self) -> Optional[str]:
        """Find the ollama executable."""
        # Check common locations
        if sys.platform == "win32":
            paths = [
                Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
                Path(os.environ.get("PROGRAMFILES", "")) / "Ollama" / "ollama.exe",
                Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe",
            ]
        else:
            paths = [
                Path("/usr/local/bin/ollama"),
                Path("/usr/bin/ollama"),
                Path.home() / ".local" / "bin" / "ollama",
                Path("/opt/homebrew/bin/ollama"),  # macOS Homebrew
            ]

        for path in paths:
            if path.exists():
                return str(path)

        # Try to find in PATH
        import shutil
        ollama_path = shutil.which("ollama")
        return ollama_path

    async def ensure_model(self, model: str) -> bool:
        """Ensure a model is available, pulling if necessary."""
        if not await self.is_running():
            return False

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:  # Long timeout for model pull
                # Check if model exists
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    models = [m.get("name", "").split(":")[0] for m in response.json().get("models", [])]
                    if model.split(":")[0] in models:
                        return True

                # Pull the model
                print(f"Pulling embedding model: {model}...")
                response = await client.post(
                    f"{self.base_url}/api/pull",
                    json={"name": model},
                    timeout=600.0  # 10 minutes for large models
                )
                return response.status_code == 200

        except Exception as e:
            print(f"Failed to ensure model {model}: {e}")
            return False

    def stop(self):
        """Stop Ollama if we started it."""
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None


class BuiltinEmbedder:
    """Embedder using sentence-transformers (no external dependencies)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._available = None

    def is_available(self) -> bool:
        """Check if sentence-transformers is installed."""
        if self._available is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def _load_model(self):
        """Lazily load the model."""
        if self._model is None and self.is_available():
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        model = self._load_model()
        if model is None:
            raise EmbeddingError("sentence-transformers not available")
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        model = self._load_model()
        if model is None:
            raise EmbeddingError("sentence-transformers not available")
        embeddings = model.encode(texts, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]


class LocalEmbedder:
    """Generate embeddings using local models (built-in or Ollama)."""

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._cache: dict[str, List[float]] = {}
        self._client: Optional[httpx.AsyncClient] = None
        self._ollama_manager = OllamaManager(self.config.ollama_base_url)
        self._builtin = BuiltinEmbedder(self.config.builtin_model)
        self._active_backend: Optional[str] = None

    async def __aenter__(self):
        await self._initialize_backend()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
        # Don't stop Ollama - let it run for other uses

    async def _initialize_backend(self):
        """Initialize the embedding backend based on config."""
        backend = self.config.backend

        if backend == "builtin" or (backend == "auto" and self._builtin.is_available()):
            if self._builtin.is_available():
                self._active_backend = "builtin"
                print("Using built-in embeddings (sentence-transformers)")
                return

        # Try Ollama
        if backend in ("ollama", "auto"):
            if self.config.auto_start_ollama:
                if await self._ollama_manager.start():
                    await self._ollama_manager.ensure_model(self.config.ollama_model)

            if await self._ollama_manager.is_running():
                self._client = httpx.AsyncClient(
                    base_url=self.config.ollama_base_url,
                    timeout=self.config.timeout_seconds
                )
                self._active_backend = "ollama"
                print("Using Ollama for embeddings")
                return

        # Fallback to built-in even if not preferred
        if self._builtin.is_available():
            self._active_backend = "builtin"
            print("Falling back to built-in embeddings")
            return

        raise EmbeddingError(
            "No embedding backend available. Install sentence-transformers or Ollama."
        )

    @property
    def active_backend(self) -> Optional[str]:
        return self._active_backend

    def _cache_key(self, text: str) -> str:
        backend = self._active_backend or "unknown"
        content = f"{backend}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def embed(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate embedding for text."""
        if self.config.cache_enabled:
            cache_key = self._cache_key(text)
            if cache_key in self._cache:
                return self._cache[cache_key]

        if self._active_backend == "builtin":
            # Run in thread pool to not block
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(None, self._builtin.embed, text)
        elif self._active_backend == "ollama":
            embedding = await self._embed_ollama(text, model)
        else:
            raise EmbeddingError("No embedding backend initialized")

        if self.config.cache_enabled:
            self._cache[self._cache_key(text)] = embedding

        return embedding

    async def _embed_ollama(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate embedding using Ollama."""
        model = model or self.config.ollama_model

        try:
            response = await self._client.post(
                "/api/embeddings",
                json={"model": model, "prompt": text}
            )
            response.raise_for_status()
            return response.json()["embedding"]

        except httpx.HTTPError as e:
            if model != self.config.ollama_fallback_model:
                return await self._embed_ollama(text, self.config.ollama_fallback_model)
            raise EmbeddingError(f"Failed to generate embedding: {e}")

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if self._active_backend == "builtin":
            # Built-in is more efficient with batching
            loop = asyncio.get_event_loop()

            # Check cache first
            uncached_texts = []
            uncached_indices = []
            results = [None] * len(texts)

            for i, text in enumerate(texts):
                if self.config.cache_enabled:
                    cache_key = self._cache_key(text)
                    if cache_key in self._cache:
                        results[i] = self._cache[cache_key]
                        continue
                uncached_texts.append(text)
                uncached_indices.append(i)

            if uncached_texts:
                embeddings = await loop.run_in_executor(
                    None, self._builtin.embed_batch, uncached_texts
                )
                for idx, embedding in zip(uncached_indices, embeddings):
                    results[idx] = embedding
                    if self.config.cache_enabled:
                        self._cache[self._cache_key(texts[idx])] = embedding

            return results

        else:
            # Ollama: process in batches
            results = []
            for i in range(0, len(texts), self.config.batch_size):
                batch = texts[i:i + self.config.batch_size]
                batch_embeddings = await asyncio.gather(
                    *[self.embed(text) for text in batch]
                )
                results.extend(batch_embeddings)
            return results

    async def check_availability(self) -> dict[str, bool]:
        """Check which backends are available."""
        availability = {
            "builtin": self._builtin.is_available(),
            "ollama": await self._ollama_manager.is_running(),
            "active": self._active_backend
        }
        return availability

    def get_status(self) -> dict:
        """Get current embedder status."""
        return {
            "active_backend": self._active_backend,
            "builtin_available": self._builtin.is_available(),
            "builtin_model": self.config.builtin_model if self._builtin.is_available() else None,
            "ollama_url": self.config.ollama_base_url,
            "cache_size": len(self._cache),
            "cache_enabled": self.config.cache_enabled
        }
