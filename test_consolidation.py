import sys
sys.path.insert(0, r"C:\dev\Nexus\Nexus\src")

def test_imports():
    """Test all module imports."""
    print("Testing imports...")
    
    # Core Nexus
    try:
        from nexus.core.ensemble_core import EnsembleCore
        print("  [OK] nexus.core")
    except Exception as e:
        print(f"  [FAIL] nexus.core: {e}")
    
    # Memory
    try:
        from nexus.memory import MemorySystem
        print("  [OK] nexus.memory")
    except Exception as e:
        print(f"  [FAIL] nexus.memory: {e}")
    
    # RAG
    try:
        from nexus.rag import RAGVectorEngine
        print("  [OK] nexus.rag")
    except Exception as e:
        print(f"  [FAIL] nexus.rag: {e}")
    
    # cog-eng
    try:
        from nexus.cog_eng import ConsciousnessCore
        from nexus.cog_eng.capabilities import AutonomousResearchAgent
        print("  [OK] nexus.cog_eng")
    except Exception as e:
        print(f"  [FAIL] nexus.cog_eng: {e}")
    
    # Providers
    try:
        from nexus.providers import MODEL_REGISTRY
        print(f"  [OK] nexus.providers ({len(MODEL_REGISTRY)} models)")
    except Exception as e:
        print(f"  [FAIL] nexus.providers: {e}")
    
    # Experts
    try:
        from nexus.experts import ConsensusEngine
        from nexus.experts.personas import ResearchExpert, AnalystExpert
        print("  [OK] nexus.experts")
    except Exception as e:
        print(f"  [FAIL] nexus.experts: {e}")
    
    # Observatory
    try:
        from nexus.observatory import MetricsCollector
        print("  [OK] nexus.observatory")
    except Exception as e:
        print(f"  [FAIL] nexus.observatory: {e}")
    
    # Insights
    try:
        from nexus.insights import InsightsEngine
        print("  [OK] nexus.insights")
    except Exception as e:
        print(f"  [FAIL] nexus.insights: {e}")
    
    # Orchestration
    try:
        from nexus.orchestration import PipelineExecutor
        print("  [OK] nexus.orchestration")
    except Exception as e:
        print(f"  [FAIL] nexus.orchestration: {e}")
    
    # Blueprints
    try:
        from nexus.blueprints import BlueprintEbookPipeline
        print("  [OK] nexus.blueprints")
    except Exception as e:
        print(f"  [FAIL] nexus.blueprints: {e}")
    
    # Platform
    try:
        from nexus.platform import NexusPlatform
        print("  [OK] nexus.platform")
    except Exception as e:
        print(f"  [FAIL] nexus.platform: {e}")
    
    print("\nImport tests completed!")


async def test_platform():
    """Test platform initialization."""
    print("\nTesting platform...")
    
    from nexus.platform import NexusPlatform
    
    platform = NexusPlatform()
    status = await platform.initialize()
    
    print("\nPlatform Status:")
    all_ok = True
    for component, ok in status.items():
        icon = "[OK]" if ok else "[FAIL]"
        print(f"  {icon} {component}")
        if not ok:
            all_ok = False
    
    if all_ok:
        print("\n[OK] Platform fully operational!")
    else:
        print("\n[PARTIAL] Some components need configuration")
    
    return all_ok


if __name__ == "__main__":
    test_imports()
    
    import asyncio
    asyncio.run(test_platform())
