"""
Tests for PyPIKnowledgeCollector — knowledge extraction from PyPI packages.
"""

import pytest
from unittest.mock import MagicMock, patch

from nexus.data.pypi_collector import (
    PyPIKnowledgeCollector,
    build_package_overview,
    build_dependency_knowledge,
    parse_dependency_names,
    extract_description_facts,
    extract_capabilities,
    _build_tags,
    DEFAULT_PACKAGES,
)
from nexus.memory.knowledge_base import KnowledgeBase, KnowledgeType


class TestBuildPackageOverview:
    """Test overview string construction."""

    def test_full_overview(self):
        result = build_package_overview(
            "torch", "2.0.0", "Tensors and neural networks", "Meta", "BSD-3", ">=3.8"
        )
        assert "torch (v2.0.0)" in result
        assert "Tensors" in result
        assert "Meta" in result

    def test_minimal_overview(self):
        result = build_package_overview("pkg", "1.0", "A package", "", "", "")
        assert "pkg" in result
        assert "A package" in result

    def test_empty_returns_empty(self):
        result = build_package_overview("x", "", "", "", "", "")
        # Too short to be useful
        assert result == "" or len(result) < 25


class TestBuildDependencyKnowledge:
    """Test dependency knowledge construction."""

    def test_with_dependencies(self):
        deps = ["numpy>=1.21", "scipy>=1.7", "pandas"]
        result = build_dependency_knowledge("torch", deps)
        assert "torch depends on" in result
        assert "numpy" in result

    def test_empty_deps(self):
        assert build_dependency_knowledge("pkg", []) == ""

    def test_with_extras(self):
        deps = [
            "numpy>=1.0",
            'torch ; extra == "gpu"',
            'jax ; extra == "tpu"',
        ]
        result = build_dependency_knowledge("ml-lib", deps)
        assert "numpy" in result
        assert "Optional extras" in result


class TestParseDependencyNames:
    """Test requirement string parsing."""

    def test_simple_deps(self):
        deps = parse_dependency_names(["numpy>=1.21", "scipy", "pandas<2.0"])
        names = [d["name"] for d in deps]
        assert "numpy" in names
        assert "scipy" in names
        assert "pandas" in names

    def test_with_extras_marker(self):
        deps = parse_dependency_names(['torch ; extra == "gpu"'])
        assert deps[0]["name"] == "torch"
        assert deps[0]["extra"] == "gpu"

    def test_no_extra(self):
        deps = parse_dependency_names(["requests>=2.28"])
        assert deps[0]["extra"] is None

    def test_empty_list(self):
        assert parse_dependency_names([]) == []

    def test_complex_version_spec(self):
        deps = parse_dependency_names(["package[extra]>=1.0,<2.0"])
        assert deps[0]["name"] == "package"


class TestExtractDescriptionFacts:
    """Test fact extraction from package descriptions."""

    def test_extracts_factual_sentences(self):
        desc = (
            "Transformers provides thousands of pretrained models. "
            "It supports PyTorch, TensorFlow, and JAX. "
            "pip install transformers. "
            "Short."
        )
        facts = extract_description_facts("transformers", desc, max_facts=5)
        assert len(facts) >= 1
        # Should skip "pip install" and "Short."
        assert not any("pip install" in f for f in facts)

    def test_limits_max_facts(self):
        desc = ". ".join(
            [f"Feature {i} is very useful for developers" for i in range(20)]
        )
        facts = extract_description_facts("pkg", desc, max_facts=3)
        assert len(facts) <= 3

    def test_empty_description(self):
        assert extract_description_facts("pkg", "", max_facts=5) == []


class TestExtractCapabilities:
    """Test classifier extraction."""

    def test_extracts_topics(self):
        classifiers = [
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Software Development :: Libraries",
            "Programming Language :: Python :: 3",
        ]
        caps = extract_capabilities(classifiers)
        assert "Artificial Intelligence" in caps
        assert "Libraries" in caps
        # Programming Language is not Topic or Framework
        assert "3" not in caps

    def test_extracts_frameworks(self):
        classifiers = ["Framework :: FastAPI", "Framework :: Flask"]
        caps = extract_capabilities(classifiers)
        assert "FastAPI" in caps
        assert "Flask" in caps

    def test_empty_classifiers(self):
        assert extract_capabilities([]) == []

    def test_limits_to_10(self):
        classifiers = [f"Topic :: Area :: Cap{i}" for i in range(20)]
        caps = extract_capabilities(classifiers)
        assert len(caps) <= 10


class TestBuildTags:
    """Test tag building."""

    def test_basic_tags(self):
        tags = _build_tags("torch", [])
        assert "pypi" in tags
        assert "torch" in tags

    def test_ml_classifier_adds_tag(self):
        tags = _build_tags("sklearn", ["Topic :: Scientific/Engineering :: Machine Learning"])
        assert "machine_learning" in tags

    def test_ai_classifier_adds_tag(self):
        tags = _build_tags("openai", ["Topic :: Scientific/Engineering :: Artificial Intelligence"])
        assert "artificial_intelligence" in tags


class TestDefaultPackages:
    """Verify the default package list."""

    def test_has_packages(self):
        assert len(DEFAULT_PACKAGES) >= 20

    def test_includes_core_ml_packages(self):
        for pkg in ("torch", "transformers", "scikit-learn", "numpy", "pandas"):
            assert pkg in DEFAULT_PACKAGES, f"Missing {pkg}"


class TestPyPICollectorInit:
    """Test collector initialization."""

    def test_init_defaults(self):
        kb = KnowledgeBase()
        collector = PyPIKnowledgeCollector(knowledge_base=kb)
        assert collector.knowledge_base is kb
        assert collector.confidence == 0.8

    def test_init_custom(self):
        kb = KnowledgeBase()
        collector = PyPIKnowledgeCollector(
            knowledge_base=kb, rate_limit_delay=0.1, confidence=0.9
        )
        assert collector.rate_limit_delay == 0.1
        assert collector.confidence == 0.9


class TestPyPICollectorCollectPackage:
    """Test single package collection with mocked HTTP."""

    @patch("nexus.data.pypi_collector.requests.get")
    def test_collect_package_stores_knowledge(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "info": {
                "name": "torch",
                "version": "2.0.0",
                "summary": "Tensors and Dynamic neural networks in Python",
                "author": "PyTorch Team",
                "license": "BSD-3",
                "requires_python": ">=3.8",
                "requires_dist": ["numpy>=1.21", "typing-extensions"],
                "classifiers": [
                    "Topic :: Scientific/Engineering :: Artificial Intelligence"
                ],
                "description": "PyTorch is an optimized tensor library for deep learning.",
            }
        }
        mock_get.return_value = mock_resp

        kb = KnowledgeBase()
        collector = PyPIKnowledgeCollector(knowledge_base=kb, rate_limit_delay=0)
        result = collector.collect_package("torch")

        assert result is not None
        assert result["items_created"] >= 3  # overview + deps + install + maybe more
        assert len(kb.knowledge_items) >= 3

    @patch("nexus.data.pypi_collector.requests.get")
    def test_collect_package_returns_none_on_404(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        kb = KnowledgeBase()
        collector = PyPIKnowledgeCollector(knowledge_base=kb, rate_limit_delay=0)
        result = collector.collect_package("nonexistent-package-xyz")
        assert result is None


class TestPyPICollectorCollectPackages:
    """Test bulk package collection."""

    @patch.object(PyPIKnowledgeCollector, "collect_package")
    def test_collect_packages_aggregates(self, mock_collect):
        mock_collect.return_value = {
            "items_created": 4,
            "knowledge_ids": ["k1", "k2", "k3", "k4"],
        }

        kb = KnowledgeBase()
        collector = PyPIKnowledgeCollector(knowledge_base=kb, rate_limit_delay=0)
        stats = collector.collect_packages(packages=["torch", "numpy"])

        assert stats["packages_processed"] == 2
        assert stats["knowledge_items_created"] == 8
        assert len(stats["knowledge_ids"]) == 8

    @patch.object(PyPIKnowledgeCollector, "collect_package")
    def test_collect_packages_handles_errors(self, mock_collect):
        mock_collect.side_effect = Exception("API down")

        kb = KnowledgeBase()
        collector = PyPIKnowledgeCollector(knowledge_base=kb, rate_limit_delay=0)
        stats = collector.collect_packages(packages=["broken"])

        assert stats["errors"] == 1
        assert stats["packages_processed"] == 0

    @patch.object(PyPIKnowledgeCollector, "collect_package")
    def test_skips_already_collected(self, mock_collect):
        mock_collect.return_value = {"items_created": 1, "knowledge_ids": ["k1"]}

        kb = KnowledgeBase()
        collector = PyPIKnowledgeCollector(knowledge_base=kb, rate_limit_delay=0)
        # First run
        collector.collect_packages(packages=["torch"])
        # Second run — should skip
        stats = collector.collect_packages(packages=["torch"])

        assert stats["skipped"] == 1
        assert stats["packages_processed"] == 0


class TestPyPICollectorStats:
    """Test collection stats."""

    def test_empty_stats(self):
        kb = KnowledgeBase()
        collector = PyPIKnowledgeCollector(knowledge_base=kb)
        stats = collector.get_collection_stats()
        assert stats["packages_collected"] == 0
        assert stats["collected_packages"] == []
