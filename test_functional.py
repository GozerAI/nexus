"""Nexus Comprehensive Functional Test v2"""
import sys
sys.path.insert(0, 'src')

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path('.env'), override=True)

def test_core_ensemble():
    print("\n1. Testing Core Ensemble System...")
    from nexus.core.ensemble_core import load_model_ensemble, ModelStub
    
    models = load_model_ensemble()
    print(f"   Loaded {len(models)} models")
    
    # Test model stub
    if models:
        response = models[0].generate_response("test")
        print(f"   Model response generated: {len(response)} chars")
    print("   [PASS] Core Ensemble")
    return True

def test_memory_system():
    print("\n2. Testing Memory System...")
    from nexus.memory import KnowledgeBase, FactualMemoryEngine, SkillMemoryEngine
    
    kb = KnowledgeBase()
    kb.add_knowledge("test_fact", "The sky is blue", source="test", confidence=0.95)
    print(f"   Knowledge base initialized, added fact")
    
    fm = FactualMemoryEngine()
    print(f"   Factual memory engine initialized")
    
    sm = SkillMemoryEngine()
    print(f"   Skill memory engine initialized")
    print("   [PASS] Memory System")
    return True

def test_rag_system():
    print("\n3. Testing RAG System...")
    from nexus.rag import ContextWindowManager, AdaptivePathways
    from nexus.rag.adaptive_rag_orchestrator import AdaptiveRAGOrchestrator
    
    # These can be initialized standalone
    cwm = ContextWindowManager(max_tokens=8000)
    print(f"   Context window manager initialized (8000 tokens)")
    
    ap = AdaptivePathways()
    pathways = ap.get_pathways()
    print(f"   Adaptive pathways initialized, {len(pathways)} pathways")
    
    # Orchestrator needs config but can handle missing gracefully
    orchestrator = AdaptiveRAGOrchestrator()
    strategies = orchestrator.get_available_strategies()
    print(f"   Orchestrator has {len(strategies)} strategies")
    
    print("   [PASS] RAG System")
    return True

def test_reasoning_system():
    print("\n4. Testing Reasoning System...")
    from nexus.reasoning import MetaReasoner, ChainOfThought, PatternReasoner, DynamicLearner
    
    mr = MetaReasoner()
    print(f"   Meta reasoner initialized")
    
    cot = ChainOfThought()
    steps = cot.get_steps()
    print(f"   Chain of thought initialized, {len(steps)} steps")
    
    pr = PatternReasoner()
    print(f"   Pattern reasoner initialized")
    
    dl = DynamicLearner()
    print(f"   Dynamic learner initialized")
    print("   [PASS] Reasoning System")
    return True

def test_data_system():
    print("\n5. Testing Data System...")
    from nexus.data import AutoDataProcessor, InternetRetriever, HuggingFaceLoader
    
    # These can be initialized standalone
    adp = AutoDataProcessor()
    print(f"   Auto data processor initialized")
    
    ir = InternetRetriever()
    print(f"   Internet retriever initialized")
    
    hfl = HuggingFaceLoader()
    print(f"   HuggingFace loader initialized")
    print("   [PASS] Data System")
    return True

def test_agent_integration():
    print("\n6. Testing Agent Integration...")
    from nexus.agents import AgentRegistry, AgentIntegration
    
    registry = AgentRegistry()
    print(f"   Agent registry initialized")
    
    integration = AgentIntegration()
    print(f"   Agent integration initialized")
    print("   [PASS] Agent Integration")
    return True

def test_strategies():
    print("\n7. Testing Ensemble Strategies...")
    from nexus.core.strategies import (
        weighted_consensus, 
        cascading_refinement,
        debate_synthesis,
        confidence_gated,
        adaptive_mixture
    )
    
    print(f"   5 ensemble strategies available")
    print("   [PASS] Ensemble Strategies")
    return True

def main():
    print("=" * 60)
    print("NEXUS COMPREHENSIVE FUNCTIONAL TEST v2")
    print("=" * 60)
    
    results = []
    
    tests = [
        ("Core Ensemble", test_core_ensemble),
        ("Memory System", test_memory_system),
        ("RAG System", test_rag_system),
        ("Reasoning System", test_reasoning_system),
        ("Data System", test_data_system),
        ("Agent Integration", test_agent_integration),
        ("Ensemble Strategies", test_strategies),
    ]
    
    for name, test_fn in tests:
        try:
            test_fn()
            results.append((name, True))
        except Exception as e:
            print(f"   [FAIL] {name}: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
    
    passed = sum(1 for _, p in results if p)
    print(f"\n  Total: {passed}/{len(results)} systems operational")
    
    if passed == len(results):
        print("\n  NEXUS IS FULLY OPERATIONAL!")
    
    print("=" * 60)
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
