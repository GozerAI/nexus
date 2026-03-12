#!/usr/bin/env python3
"""
Test basic imports for all Nexus modules.
"""

import sys
import traceback

def test_import(module_name, description):
    """Test importing a module."""
    try:
        __import__(module_name)
        print(f"[OK] {description:40s}")
        return True
    except Exception as e:
        print(f"[FAIL] {description:40s}")
        print(f"  Error: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Run all import tests."""
    print("=" * 80)
    print("NEXUS MODULE IMPORT TESTS")
    print("=" * 80)
    print()

    tests = [
        # Core modules
        ("nexus.core", "Core Ensemble System"),
        ("nexus.core.ensemble_core", "Ensemble Core"),
        ("nexus.core.auth", "Authentication"),
        ("nexus.core.cache", "Caching"),
        ("nexus.core.monitoring", "Monitoring"),

        # Memory modules
        ("nexus.memory", "Memory System"),
        ("nexus.memory.knowledge_base", "Knowledge Base"),
        ("nexus.memory.factual_memory_engine", "Factual Memory"),
        ("nexus.memory.skill_memory_engine", "Skill Memory"),
        ("nexus.memory.pattern_recognition_engine", "Pattern Recognition"),
        ("nexus.memory.memory_block_manager", "Memory Block Manager"),
        ("nexus.memory.knowledge_validator", "Knowledge Validator"),
        ("nexus.memory.knowledge_gap_tracker", "Knowledge Gap Tracker"),
        ("nexus.memory.knowledge_expander", "Knowledge Expander"),
        ("nexus.memory.memory_analytics", "Memory Analytics"),
        ("nexus.memory.knowledge_graph_visualizer", "Knowledge Graph Visualizer"),

        # RAG modules
        ("nexus.rag", "RAG System"),
        ("nexus.rag.rag_vector_engine", "RAG Vector Engine"),
        ("nexus.rag.adaptive_rag_orchestrator", "Adaptive RAG Orchestrator"),
        ("nexus.rag.context_window_manager", "Context Window Manager"),
        ("nexus.rag.adaptive_pathways", "Adaptive Pathways"),

        # Reasoning modules
        ("nexus.reasoning", "Reasoning System"),
        ("nexus.reasoning.meta_reasoner", "Meta Reasoner"),
        ("nexus.reasoning.chain_of_thought", "Chain of Thought"),
        ("nexus.reasoning.pattern_reasoner", "Pattern Reasoner"),
        ("nexus.reasoning.dynamic_learner", "Dynamic Learner"),
        ("nexus.reasoning.analytics", "Reasoning Analytics"),

        # Data modules
        ("nexus.data", "Data System"),
        ("nexus.data.data_ingestion", "Data Ingestion"),
        ("nexus.data.auto_data_processor", "Auto Data Processor"),
        ("nexus.data.internet_retriever", "Internet Retriever"),
        ("nexus.data.huggingface_loader", "HuggingFace Loader"),

        # API modules
        ("nexus.api", "API System"),

        # Agent modules
        ("nexus.agents", "Agent Integration"),

        # UI modules
        ("nexus.ui", "UI System"),
    ]

    print("Testing module imports...")
    print("-" * 80)

    passed = 0
    failed = 0

    for module_name, description in tests:
        if test_import(module_name, description):
            passed += 1
        else:
            failed += 1
        print()

    print("-" * 80)
    print(f"\nResults: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 80)

    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
