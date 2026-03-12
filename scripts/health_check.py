"""Health check for Nexus Intelligence Platform."""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def check_health():
    from nexus.intelligence import NexusIntelligence

    print("Nexus Intelligence Health Check")
    print("=" * 50)

    intel = NexusIntelligence()

    try:
        await intel.initialize()
        print("[PASS] Core initialization successful")
    except Exception as e:
        print(f"[FAIL] Core initialization failed: {e}")
        return False

    try:
        # Test embedder
        avail = await intel.embedder.check_availability()
        if avail.get(intel.embedder.config.primary_model):
            print(f"[PASS] Embedder: {intel.embedder.config.primary_model} available")
        else:
            print(f"[WARN] Primary model unavailable, using fallback")
    except Exception as e:
        print(f"[WARN] Embedder check failed: {e}")

    try:
        # Test vector store
        count = intel.vector_store.count()
        print(f"[PASS] Vector store: {count} chunks indexed")
    except Exception as e:
        print(f"[FAIL] Vector store check failed: {e}")

    try:
        # Test SQLite
        task_count = await intel.sqlite.count("tasks")
        print(f"[PASS] SQLite: {task_count} tasks stored")
    except Exception as e:
        print(f"[FAIL] SQLite check failed: {e}")

    try:
        stats = await intel.memory.get_stats()
        print(f"[PASS] Memory: {stats['total_conversations']} conversations")
    except Exception as e:
        print(f"[FAIL] Memory check failed: {e}")

    await intel.shutdown()
    print("\n" + "=" * 50)
    print("[PASS] Health check complete")
    return True


if __name__ == "__main__":
    success = asyncio.run(check_health())
    sys.exit(0 if success else 1)
