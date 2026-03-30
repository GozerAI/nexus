"""
Offline Model Inference with Quantized Models (Item 780)

Provides local model inference using quantized models for offline operation.
Supports GGUF/GGML formats via ctransformers/llama-cpp-python (optional deps),
and a built-in simple completion engine for basic tasks.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class QuantizationType(str, Enum):
    """Quantization format types."""

    GGUF = "gguf"
    GGML = "ggml"
    GPTQ = "gptq"
    AWQ = "awq"
    NATIVE = "native"  # Full-precision local model


class InferenceBackend(str, Enum):
    """Available inference backends."""

    LLAMA_CPP = "llama_cpp"
    CTRANSFORMERS = "ctransformers"
    OLLAMA_LOCAL = "ollama_local"
    SIMPLE = "simple"  # Built-in pattern-based completion


@dataclass
class LocalModel:
    """Registration of a local model."""

    model_id: str
    name: str
    path: Optional[str] = None  # Path to model file
    backend: InferenceBackend = InferenceBackend.SIMPLE
    quantization: QuantizationType = QuantizationType.NATIVE
    context_length: int = 2048
    parameters: Dict[str, Any] = field(default_factory=dict)
    loaded: bool = False
    avg_tokens_per_second: float = 0.0
    total_inferences: int = 0


@dataclass
class InferenceRequest:
    """A request for local model inference."""

    prompt: str
    model_id: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: List[str] = field(default_factory=list)


@dataclass
class InferenceResult:
    """Result of a local model inference."""

    model_id: str
    content: str
    tokens_generated: int
    latency_ms: float
    tokens_per_second: float
    backend: InferenceBackend
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimpleCompletionEngine:
    """
    A basic pattern-matching completion engine for offline fallback.
    Uses template-based responses for common task patterns.
    """

    def __init__(self):
        self._templates: Dict[str, str] = {
            "summarize": "Summary: {input_excerpt}",
            "classify": "Classification: Based on the input, the category is general.",
            "extract": "Extracted information: {input_excerpt}",
            "translate": "Translation: [Offline - translation not available]",
            "default": "Response: Based on the available information, {input_excerpt}",
        }
        self._custom_handlers: Dict[str, Callable] = {}

    def register_handler(self, task_pattern: str, handler: Callable[[str], str]) -> None:
        self._custom_handlers[task_pattern] = handler

    def complete(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate a simple completion based on prompt patterns."""
        prompt_lower = prompt.lower()

        # Check custom handlers first
        for pattern, handler in self._custom_handlers.items():
            if pattern in prompt_lower:
                return handler(prompt)

        # Template matching
        excerpt = prompt[:200].strip()
        if "summar" in prompt_lower:
            return self._templates["summarize"].format(input_excerpt=excerpt)
        elif "classif" in prompt_lower or "categoriz" in prompt_lower:
            return self._templates["classify"].format(input_excerpt=excerpt)
        elif "extract" in prompt_lower:
            return self._templates["extract"].format(input_excerpt=excerpt)
        elif "translat" in prompt_lower:
            return self._templates["translate"].format(input_excerpt=excerpt)
        else:
            return self._templates["default"].format(input_excerpt=excerpt)


class OfflineModelInference:
    """
    Manages local model inference for offline operation.

    Supports:
    - Registration of local quantized models
    - Multiple backend support (llama.cpp, ctransformers, ollama local)
    - Built-in simple completion engine as last-resort fallback
    - Performance tracking per model
    - Model selection based on task requirements
    """

    def __init__(self):
        self._models: Dict[str, LocalModel] = {}
        self._backends: Dict[InferenceBackend, Any] = {}
        self._simple_engine = SimpleCompletionEngine()
        self._inference_history: deque = deque(maxlen=5000)
        self._backend_loaders: Dict[InferenceBackend, Callable] = {}

    # ── Model Registration ──────────────────────────────────────────

    def register_model(
        self,
        model_id: str,
        name: str,
        path: Optional[str] = None,
        backend: InferenceBackend = InferenceBackend.SIMPLE,
        quantization: QuantizationType = QuantizationType.NATIVE,
        context_length: int = 2048,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> LocalModel:
        """Register a local model for inference."""
        model = LocalModel(
            model_id=model_id,
            name=name,
            path=path,
            backend=backend,
            quantization=quantization,
            context_length=context_length,
            parameters=parameters or {},
        )
        self._models[model_id] = model
        logger.info("Registered local model: %s (%s)", name, backend.value)
        return model

    def register_backend_loader(
        self, backend: InferenceBackend, loader: Callable
    ) -> None:
        """Register a callable that loads and returns a backend instance."""
        self._backend_loaders[backend] = loader

    def register_completion_handler(
        self, pattern: str, handler: Callable[[str], str]
    ) -> None:
        """Register a custom handler for the simple completion engine."""
        self._simple_engine.register_handler(pattern, handler)

    # ── Inference ───────────────────────────────────────────────────

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """
        Run inference on a local model.
        Falls back to the simple engine if the requested model/backend
        is not available.
        """
        model = self._models.get(request.model_id)
        start = time.time()

        if model is None:
            # Fall back to simple engine
            content = self._simple_engine.complete(request.prompt, request.max_tokens)
            latency = (time.time() - start) * 1000
            result = InferenceResult(
                model_id="simple_engine",
                content=content,
                tokens_generated=len(content) // 4,
                latency_ms=latency,
                tokens_per_second=len(content) // 4 / max(latency / 1000, 0.001),
                backend=InferenceBackend.SIMPLE,
                metadata={"fallback": True, "reason": "model_not_found"},
            )
            self._inference_history.append(result)
            return result

        # Try the model's backend
        try:
            content = self._run_backend_inference(model, request)
        except Exception as e:
            logger.warning(
                "Backend %s failed for %s: %s, falling back to simple engine",
                model.backend.value,
                model.model_id,
                e,
            )
            content = self._simple_engine.complete(request.prompt, request.max_tokens)

        latency = (time.time() - start) * 1000
        tokens = len(content) // 4
        tps = tokens / max(latency / 1000, 0.001)

        model.total_inferences += 1
        model.avg_tokens_per_second = (
            model.avg_tokens_per_second * (model.total_inferences - 1) + tps
        ) / model.total_inferences

        result = InferenceResult(
            model_id=model.model_id,
            content=content,
            tokens_generated=tokens,
            latency_ms=latency,
            tokens_per_second=tps,
            backend=model.backend,
        )
        self._inference_history.append(result)
        return result

    def _run_backend_inference(
        self, model: LocalModel, request: InferenceRequest
    ) -> str:
        """Run inference using the model's backend."""
        if model.backend == InferenceBackend.SIMPLE:
            return self._simple_engine.complete(request.prompt, request.max_tokens)

        # Check if we have a loader for this backend
        loader = self._backend_loaders.get(model.backend)
        if loader:
            if model.backend not in self._backends:
                self._backends[model.backend] = loader(model)
                model.loaded = True

            backend = self._backends[model.backend]
            # Generic interface: backend must have a __call__ or generate method
            if callable(backend):
                return backend(request.prompt, max_tokens=request.max_tokens)
            elif hasattr(backend, "generate"):
                return backend.generate(request.prompt, max_tokens=request.max_tokens)

        # If no specific backend, fall back
        return self._simple_engine.complete(request.prompt, request.max_tokens)

    def select_model(
        self,
        task_type: Optional[str] = None,
        max_latency_ms: Optional[float] = None,
    ) -> Optional[str]:
        """Select the best available local model for a task."""
        candidates = list(self._models.values())

        if max_latency_ms and any(m.total_inferences > 0 for m in candidates):
            candidates = [
                m
                for m in candidates
                if m.total_inferences == 0
                or (m.avg_tokens_per_second * max_latency_ms / 1000) > 10
            ]

        if not candidates:
            return None

        # Prefer models with highest throughput
        candidates.sort(key=lambda m: m.avg_tokens_per_second, reverse=True)
        return candidates[0].model_id

    # ── Reporting ───────────────────────────────────────────────────

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "model_id": m.model_id,
                "name": m.name,
                "backend": m.backend.value,
                "quantization": m.quantization.value,
                "loaded": m.loaded,
                "total_inferences": m.total_inferences,
                "avg_tokens_per_second": round(m.avg_tokens_per_second, 1),
            }
            for m in self._models.values()
        ]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "registered_models": len(self._models),
            "loaded_backends": list(
                b.value for b in self._backends.keys()
            ),
            "total_inferences": len(self._inference_history),
            "models": self.list_models(),
        }
