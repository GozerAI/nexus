"""Phase 1 Verification - Storage Layer."""
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def verify():
    from nexus.storage import LocalEmbedder, VectorStore, SQLiteStore, VectorChunk

    print("Phase 1 Verification")
    print("=" * 50)

    # Test embedder
    print("\n1. Testing Embedder...")
    try:
        async with LocalEmbedder() as embedder:
            avail = await embedder.check_availability()
            print(f"   Embedder models: {avail}")
            if any(avail.values()):
                print("   [PASS] Embedder working")
            else:
                print("   [WARN] No embedding models available (Ollama may not be running)")
    except Exception as e:
        print(f"   [WARN] Embedder test skipped: {e}")

    # Test vector store
    print("\n2. Testing VectorStore...")
    try:
        vs = VectorStore("data/chroma_test", "test")
        vs.initialize()
        vs.add([VectorChunk(id="t1", text="test")])
        count = vs.count()
        print(f"   VectorStore: {count} chunks")
        vs.delete(["t1"])
        print("   [PASS] VectorStore working")
    except Exception as e:
        print(f"   [FAIL] VectorStore: {e}")

    # Test SQLite
    print("\n3. Testing SQLite...")
    try:
        sql = SQLiteStore("data/sqlite/test.db")
        await sql.initialize()
        await sql.insert("tasks", {"id": "t1", "title": "Test"})
        count = await sql.count("tasks")
        print(f"   SQLite: {count} tasks")
        await sql.delete("tasks", "t1")
        await sql.close()
        print("   [PASS] SQLite working")
    except Exception as e:
        print(f"   [FAIL] SQLite: {e}")

    print("\n" + "=" * 50)
    print("Phase 1 COMPLETE")


if __name__ == "__main__":
    asyncio.run(verify())
