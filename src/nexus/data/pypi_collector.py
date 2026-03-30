"""
PyPI Knowledge Collector for Nexus AI Platform.

Extracts knowledge from Python package metadata and stores it
in the KnowledgeBase. Complements the existing PyPIIntegration
(discovery layer) by focusing on knowledge extraction:

- Package descriptions → FACTUAL knowledge
- Dependency relationships → FACTUAL knowledge
- Installation / usage patterns → PROCEDURAL knowledge
- Capability summaries → FACTUAL knowledge
"""

import logging
import time
from typing import Any, Dict, List, Optional, Set

import requests

from nexus.memory.knowledge_base import KnowledgeBase, KnowledgeType

logger = logging.getLogger(__name__)

# AI/ML packages to collect knowledge about (superset of discovery list)
DEFAULT_PACKAGES = [
    "transformers",
    "torch",
    "tensorflow",
    "langchain",
    "openai",
    "anthropic",
    "huggingface-hub",
    "sentence-transformers",
    "accelerate",
    "peft",
    "datasets",
    "evaluate",
    "gradio",
    "streamlit",
    "fastapi",
    "chromadb",
    "pinecone-client",
    "weaviate-client",
    "llama-index",
    "autogen",
    "crewai",
    "instructor",
    "outlines",
    "vllm",
    "ollama",
    "scikit-learn",
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "seaborn",
    "plotly",
    "keras",
    "jax",
    "flax",
    "optax",
    "ray",
    "dask",
    "polars",
    "xgboost",
    "lightgbm",
    "catboost",
    "spacy",
    "nltk",
    "gensim",
    "flair",
    "onnxruntime",
    "triton",
    "mlflow",
    "wandb",
]


class PyPIKnowledgeCollector:
    """
    Collects knowledge from PyPI package metadata.

    Fetches package info, extracts structured knowledge items,
    and stores them in the KnowledgeBase for retrieval.
    """

    API_BASE = "https://pypi.org/pypi"

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        rate_limit_delay: float = 0.5,
        confidence: float = 0.8,
    ):
        self.knowledge_base = knowledge_base
        self.rate_limit_delay = rate_limit_delay
        self.confidence = confidence
        self._last_request_time = 0.0
        self._collected_packages: Set[str] = set()

    def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def collect_packages(
        self,
        packages: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Collect knowledge from PyPI for a list of packages.

        Args:
            packages: Package names. Defaults to AI/ML packages.

        Returns:
            Collection statistics.
        """
        packages = packages or DEFAULT_PACKAGES
        stats = {
            "packages_processed": 0,
            "knowledge_items_created": 0,
            "knowledge_ids": [],
            "errors": 0,
            "skipped": 0,
        }

        for package_name in packages:
            if package_name in self._collected_packages:
                stats["skipped"] += 1
                continue

            try:
                result = self.collect_package(package_name)
                if result:
                    stats["packages_processed"] += 1
                    stats["knowledge_items_created"] += result["items_created"]
                    stats["knowledge_ids"].extend(result["knowledge_ids"])
                    self._collected_packages.add(package_name)
                else:
                    stats["errors"] += 1
            except Exception as e:
                logger.error(f"Error collecting package '{package_name}': {e}")
                stats["errors"] += 1

        logger.info(
            f"PyPI collection complete: {stats['packages_processed']} packages, "
            f"{stats['knowledge_items_created']} knowledge items"
        )
        return stats

    def fetch_package_info(self, package_name: str) -> Optional[Dict[str, Any]]:
        """Fetch package metadata from PyPI JSON API."""
        self._rate_limit()
        url = f"{self.API_BASE}/{package_name}/json"
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                logger.debug(f"Package not found: {package_name}")
            else:
                logger.warning(f"PyPI returned {response.status_code} for {package_name}")
        except Exception as e:
            logger.error(f"Failed to fetch {package_name}: {e}")
        return None

    def collect_package(self, package_name: str) -> Optional[Dict[str, Any]]:
        """
        Collect knowledge from a single PyPI package.

        Extracts:
        - Package summary as factual knowledge
        - Full description sections as factual knowledge
        - Dependency info as factual knowledge
        - Install/usage as procedural knowledge

        Returns:
            Dict with items_created and knowledge_ids, or None on failure.
        """
        data = self.fetch_package_info(package_name)
        if not data:
            return None

        info = data.get("info", {})
        result = {"items_created": 0, "knowledge_ids": []}

        name = info.get("name", package_name)
        summary = info.get("summary", "")
        version = info.get("version", "")
        description = info.get("description", "")
        requires_python = info.get("requires_python", "")
        requires_dist = info.get("requires_dist") or []
        license_name = info.get("license", "")
        author = info.get("author", "")
        classifiers = info.get("classifiers", [])

        tags = _build_tags(name, classifiers)

        # 1. Package overview (factual)
        overview = build_package_overview(
            name, version, summary, author, license_name, requires_python
        )
        if overview:
            kid = self.knowledge_base.add_knowledge(
                content=overview,
                knowledge_type=KnowledgeType.FACTUAL,
                source=f"pypi:{name}",
                confidence=self.confidence,
                context_tags=tags,
            )
            result["knowledge_ids"].append(kid)
            result["items_created"] += 1

        # 2. Dependencies as knowledge (factual)
        dep_knowledge = build_dependency_knowledge(name, requires_dist)
        if dep_knowledge:
            kid = self.knowledge_base.add_knowledge(
                content=dep_knowledge,
                knowledge_type=KnowledgeType.FACTUAL,
                source=f"pypi:{name}:dependencies",
                confidence=self.confidence,
                context_tags=tags + ["dependencies"],
            )
            result["knowledge_ids"].append(kid)
            result["items_created"] += 1

        # 3. Install instructions (procedural)
        install_knowledge = f"To install {name}: pip install {name}"
        if requires_python:
            install_knowledge += f" (requires Python {requires_python})"
        kid = self.knowledge_base.add_knowledge(
            content=install_knowledge,
            knowledge_type=KnowledgeType.PROCEDURAL,
            source=f"pypi:{name}:install",
            confidence=0.95,  # Install commands are highly reliable
            context_tags=tags + ["installation"],
        )
        result["knowledge_ids"].append(kid)
        result["items_created"] += 1

        # 4. Extract facts from description (if available and not too long)
        if description:
            desc_facts = extract_description_facts(
                name, description, max_facts=5
            )
            for fact in desc_facts:
                kid = self.knowledge_base.add_knowledge(
                    content=fact,
                    knowledge_type=KnowledgeType.FACTUAL,
                    source=f"pypi:{name}:description",
                    confidence=self.confidence - 0.05,
                    context_tags=tags,
                )
                result["knowledge_ids"].append(kid)
                result["items_created"] += 1

        # 5. Capability classification from classifiers (factual)
        capabilities = extract_capabilities(classifiers)
        if capabilities:
            cap_text = f"{name} is classified as: {', '.join(capabilities)}."
            kid = self.knowledge_base.add_knowledge(
                content=cap_text,
                knowledge_type=KnowledgeType.FACTUAL,
                source=f"pypi:{name}:classifiers",
                confidence=self.confidence,
                context_tags=tags + ["capabilities"],
            )
            result["knowledge_ids"].append(kid)
            result["items_created"] += 1

        if result["items_created"] > 0:
            logger.info(f"Collected {result['items_created']} items from '{name}'")

        return result

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return statistics about collected packages."""
        return {
            "packages_collected": len(self._collected_packages),
            "collected_packages": sorted(self._collected_packages),
        }


def build_package_overview(
    name: str,
    version: str,
    summary: str,
    author: str,
    license_name: str,
    requires_python: str,
) -> str:
    """Build a concise package overview string."""
    parts = [f"{name} (v{version})" if version else name]
    if summary:
        parts.append(f"is a Python package: {summary}")
    if author:
        parts.append(f"Author: {author}")
    if license_name and len(license_name) < 50:
        parts.append(f"License: {license_name}")
    if requires_python:
        parts.append(f"Requires Python {requires_python}")
    return ". ".join(parts) + "." if len(". ".join(parts)) >= 20 else ""


def build_dependency_knowledge(name: str, requires_dist: List[str]) -> str:
    """Build knowledge about package dependencies."""
    if not requires_dist:
        return ""

    # Parse just the package names (strip version specs and extras)
    deps = parse_dependency_names(requires_dist)
    core_deps = [d for d in deps if d["extra"] is None]
    optional_deps = [d for d in deps if d["extra"] is not None]

    parts = []
    if core_deps:
        dep_names = [d["name"] for d in core_deps[:15]]
        parts.append(f"{name} depends on: {', '.join(dep_names)}")
    if optional_deps:
        extras = set(d["extra"] for d in optional_deps)
        parts.append(f"Optional extras: {', '.join(sorted(extras)[:10])}")

    return ". ".join(parts) + "." if parts else ""


def parse_dependency_names(requires_dist: List[str]) -> List[Dict[str, Optional[str]]]:
    """
    Parse requirement strings into package name + optional extra.

    Examples:
        "numpy>=1.21" → {"name": "numpy", "extra": None}
        "torch ; extra == 'gpu'" → {"name": "torch", "extra": "gpu"}
    """
    results = []
    for req in requires_dist:
        # Split on semicolon for markers
        parts = req.split(";")
        name_part = parts[0].strip()
        # Extract package name (before version specifiers)
        pkg_name = ""
        for ch in name_part:
            if ch in (">", "<", "=", "!", "[", " ", "~"):
                break
            pkg_name += ch
        pkg_name = pkg_name.strip()

        extra = None
        if len(parts) > 1:
            marker = parts[1].strip()
            if "extra" in marker:
                # Extract extra name from marker like: extra == "gpu"
                for q in ('"', "'"):
                    if q in marker:
                        start = marker.index(q) + 1
                        end = marker.index(q, start)
                        extra = marker[start:end]
                        break

        if pkg_name:
            results.append({"name": pkg_name, "extra": extra})
    return results


def extract_description_facts(
    package_name: str, description: str, max_facts: int = 5
) -> List[str]:
    """Extract factual sentences from a package description."""
    # Take first 2000 chars to avoid processing huge READMEs
    text = description[:2000]

    # Strip markdown-style formatting
    import re
    text = re.sub(r"[#*`~\[\]()]", " ", text)
    text = re.sub(r"!\[.*?\]", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text)

    sentences = text.split(". ")
    facts = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 30 or len(sentence) > 300:
            continue
        lower = sentence.lower()
        # Must be a factual assertion
        if not any(w in lower for w in ("is", "are", "provides", "supports", "enables", "allows")):
            continue
        # Skip install/command instructions (those are procedural, handled separately)
        if any(w in lower for w in ("pip install", "$ ", ">>>", "import ")):
            continue
        facts.append(sentence)
        if len(facts) >= max_facts:
            break

    return facts


def extract_capabilities(classifiers: List[str]) -> List[str]:
    """Extract human-readable capabilities from PyPI classifiers."""
    capabilities = []
    for classifier in classifiers:
        parts = classifier.split(" :: ")
        if len(parts) >= 3 and parts[0] == "Topic":
            cap = parts[-1]
            if cap not in capabilities:
                capabilities.append(cap)
        elif len(parts) >= 2 and parts[0] == "Framework":
            cap = parts[-1]
            if cap not in capabilities:
                capabilities.append(cap)
    return capabilities[:10]


def _build_tags(name: str, classifiers: List[str]) -> List[str]:
    """Build context tags for PyPI knowledge items."""
    tags = ["pypi", name.lower()]
    for classifier in classifiers:
        if "Machine Learning" in classifier:
            tags.append("machine_learning")
            break
        if "Artificial Intelligence" in classifier:
            tags.append("artificial_intelligence")
            break
    return tags
