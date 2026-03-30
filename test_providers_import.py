import sys
sys.path.insert(0, r"C:\dev\Nexus\Nexus\src")

try:
    from nexus.providers import MODEL_REGISTRY
    print(f"[OK] Providers imported, {len(MODEL_REGISTRY)} models registered")
except ImportError as e:
    print(f"[FAIL] Import error: {e}")
except Exception as e:
    print(f"[FAIL] Other error: {e}")
