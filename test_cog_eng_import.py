import sys
sys.path.insert(0, r"C:\dev\Nexus\Nexus\src")

try:
    from nexus.cog_eng import ConsciousnessCore, AutonomousResearchAgent
    print("[OK] cog-eng imported successfully")
except ImportError as e:
    print(f"[FAIL] Import error: {e}")
