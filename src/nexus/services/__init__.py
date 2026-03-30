"""
Nexus Services - Background services for the intelligence platform.
"""

from .ollama_manager import (
    OllamaManager,
    OllamaStatus,
    get_ollama_manager,
    ensure_ollama_running,
)

__all__ = [
    "OllamaManager",
    "OllamaStatus",
    "get_ollama_manager",
    "ensure_ollama_running",
]
