"""Nexus System Status Check"""
import sys
sys.path.insert(0, 'src')

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path('.env'), override=True)

print("=" * 60)
print("NEXUS SYSTEM STATUS CHECK")
print("=" * 60)

# Track results
passed = 0
total = 0

def check(name, fn):
    global passed, total
    total += 1
    try:
        fn()
        print(f"  [OK] {name}")
        passed += 1
        return True
    except Exception as e:
        print(f"  [--] {name}: {str(e)[:50]}")
        return False

print("\n--- Core Systems ---")
check("Ensemble Core", lambda: __import__('nexus.core.ensemble_core'))
check("Core Engine", lambda: __import__('nexus.core.core_engine'))
check("Strategies (19K lines)", lambda: __import__('nexus.core.strategies'))
check("Config Validator", lambda: __import__('nexus.core.config_validator'))

print("\n--- Memory Systems (130K lines) ---")
check("Knowledge Base", lambda: __import__('nexus.memory.knowledge_base'))
check("Factual Memory Engine", lambda: __import__('nexus.memory.factual_memory_engine'))
check("Skill Memory Engine", lambda: __import__('nexus.memory.skill_memory_engine'))
check("Pattern Recognition", lambda: __import__('nexus.memory.pattern_recognition_engine'))
check("Memory Block Manager", lambda: __import__('nexus.memory.memory_block_manager'))
check("Knowledge Validator", lambda: __import__('nexus.memory.knowledge_validator'))
check("Knowledge Gap Tracker", lambda: __import__('nexus.memory.knowledge_gap_tracker'))
check("Knowledge Expander", lambda: __import__('nexus.memory.knowledge_expander'))
check("Memory Analytics", lambda: __import__('nexus.memory.memory_analytics'))
check("Knowledge Graph Visualizer", lambda: __import__('nexus.memory.knowledge_graph_visualizer'))

print("\n--- RAG Systems ---")
check("RAG Vector Engine", lambda: __import__('nexus.rag.rag_vector_engine'))
check("Adaptive RAG Orchestrator", lambda: __import__('nexus.rag.adaptive_rag_orchestrator'))
check("Context Window Manager", lambda: __import__('nexus.rag.context_window_manager'))
check("Adaptive Pathways", lambda: __import__('nexus.rag.adaptive_pathways'))

print("\n--- Reasoning Systems ---")
check("Meta Reasoner", lambda: __import__('nexus.reasoning.meta_reasoner'))
check("Chain of Thought", lambda: __import__('nexus.reasoning.chain_of_thought'))
check("Pattern Reasoner", lambda: __import__('nexus.reasoning.pattern_reasoner'))
check("Dynamic Learner", lambda: __import__('nexus.reasoning.dynamic_learner'))
check("Reasoning Analytics", lambda: __import__('nexus.reasoning.analytics'))

print("\n--- Data Systems ---")
check("Data Ingestion", lambda: __import__('nexus.data.data_ingestion'))
check("Auto Data Processor", lambda: __import__('nexus.data.auto_data_processor'))
check("Internet Retriever", lambda: __import__('nexus.data.internet_retriever'))
check("HuggingFace Loader", lambda: __import__('nexus.data.huggingface_loader'))

print("\n--- API & Integration ---")
check("API System", lambda: __import__('nexus.api.api'))
check("Agent Registry", lambda: __import__('nexus.agents.registry'))

print("\n--- Quick Functional Tests ---")

def test_models():
    from nexus.core.ensemble_core import load_model_ensemble
    models = load_model_ensemble()
    assert len(models) == 4
    return True
check("Load 4 Models", test_models)

def test_kb():
    from nexus.memory.knowledge_base import KnowledgeBase
    kb = KnowledgeBase()
    return True
check("Init Knowledge Base", test_kb)

def test_reasoner():
    from nexus.reasoning.meta_reasoner import MetaReasoner
    mr = MetaReasoner()
    return True
check("Init Meta Reasoner", test_reasoner)

print("\n" + "=" * 60)
print(f"TOTAL: {passed}/{total} checks passed ({100*passed//total}%)")
print("=" * 60)

if passed >= total * 0.9:
    print("\nNEXUS STATUS: OPERATIONAL")
else:
    print("\nNEXUS STATUS: NEEDS ATTENTION")
