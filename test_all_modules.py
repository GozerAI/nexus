import sys
sys.path.insert(0, r"C:\dev\Nexus\Nexus\src")

modules = [
    ("nexus.experts", "ConsensusEngine"),
    ("nexus.observatory", "MetricsCollector"),
    ("nexus.insights", "InsightsEngine"),
    ("nexus.orchestration", "PipelineExecutor"),
    ("nexus.blueprints", "BlueprintEbookPipeline"),
]

for module_name, class_name in modules:
    try:
        mod = __import__(module_name, fromlist=[class_name])
        print(f"[OK] {module_name}")
    except ImportError as e:
        print(f"[FAIL] {module_name}: {e}")
    except Exception as e:
        print(f"[WARN] {module_name}: {e}")
