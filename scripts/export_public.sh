#!/usr/bin/env bash
# export_public.sh — Creates a clean public export of Nexus for GozerAI/nexus.
# Usage: bash scripts/export_public.sh [target_dir]
#
# Strips proprietary Pro/Enterprise modules and internal infrastructure,
# leaving only community-tier code + the license gate (so users see the upgrade path).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${1:-${REPO_ROOT}/../nexus-public-export}"

echo "=== Nexus Public Export ==="
echo "Source: ${REPO_ROOT}"
echo "Target: ${TARGET}"

# Clean target
rm -rf "${TARGET}"
mkdir -p "${TARGET}"

# Use git archive to get a clean copy (respects .gitignore, excludes .git)
cd "${REPO_ROOT}"
git archive HEAD | tar -x -C "${TARGET}"

# ===== STRIP PROPRIETARY MODULES (Pro/Enterprise) =====

# Pro tier — ensemble & reasoning internals
rm -rf "${TARGET}/src/nexus/providers/ensemble/"
rm -rf "${TARGET}/src/nexus/providers/strategies/"
rm -f  "${TARGET}/src/nexus/core/ensemble_core.py"
rm -f  "${TARGET}/src/nexus/core/ensemble_core_v2.py"
rm -f  "${TARGET}/src/nexus/core/strategic_ensemble.py"
rm -rf "${TARGET}/src/nexus/reasoning/"

# Enterprise tier — discovery, intelligence, cognitive engineering
rm -rf "${TARGET}/src/nexus/discovery/"
rm -rf "${TARGET}/src/nexus/intelligence/"
rm -rf "${TARGET}/src/nexus/insights/"
rm -rf "${TARGET}/src/nexus/cog_eng/"
rm -rf "${TARGET}/src/nexus/experts/"
rm -rf "${TARGET}/src/nexus/data/"
rm -rf "${TARGET}/src/nexus/automations/"
rm -rf "${TARGET}/src/nexus/blueprints/"

# Internal / private — C-Suite bridge, COO, GUI, UI
rm -rf "${TARGET}/src/nexus/csuite/"
rm -rf "${TARGET}/src/nexus/coo/"
rm -rf "${TARGET}/src/nexus/gui/"
rm -rf "${TARGET}/src/nexus/ui/"

# ===== STRIP INFRASTRUCTURE =====
rm -rf "${TARGET}/infrastructure/helm/"
rm -rf "${TARGET}/infrastructure/kubernetes/"
rm -f  "${TARGET}/config/config_enhanced.yaml"
rm -f  "${TARGET}/config/nexus_intelligence.yaml"
rm -f  "${TARGET}/config/simulation_config.yaml"

# ===== STRIP TESTS FOR PROPRIETARY MODULES =====
rm -f  "${TARGET}/tests/test_ensemble_core.py"
rm -f  "${TARGET}/tests/test_enhanced_api.py"
rm -f  "${TARGET}/tests/unit/test_reasoning.py"
rm -f  "${TARGET}/tests/unit/test_data.py"
rm -f  "${TARGET}/tests/unit/test_zuultimate_integration.py"
rm -f  "${TARGET}/tests/unit/test_vinzy_integration.py"
rm -f  "${TARGET}/tests/unit/test_kh_graph_collector.py"
rm -f  "${TARGET}/tests/unit/test_workflow_collector.py"
rm -f  "${TARGET}/tests/unit/test_wikipedia_collector.py"
rm -f  "${TARGET}/tests/unit/test_pypi_collector.py"

# ===== CREATE STUB __init__.py FOR STRIPPED PACKAGES =====
# So imports fail gracefully with clear "requires Pro/Enterprise" messages

for pkg in reasoning discovery intelligence insights cog_eng experts data automations blueprints; do
    mkdir -p "${TARGET}/src/nexus/${pkg}"
    cat > "${TARGET}/src/nexus/${pkg}/__init__.py" << 'PYEOF'
"""This module requires a commercial license.

Visit https://gozerai.com/pricing for Pro and Enterprise tier details.
Set VINZY_LICENSE_KEY to unlock licensed features.
"""

raise ImportError(
    f"{__name__} requires a commercial Nexus license. "
    "Visit https://gozerai.com/pricing for details."
)
PYEOF
done

# Ensemble stub (nested under providers)
mkdir -p "${TARGET}/src/nexus/providers/ensemble"
cat > "${TARGET}/src/nexus/providers/ensemble/__init__.py" << 'PYEOF'
"""Multi-model ensemble requires a Pro license.

Visit https://gozerai.com/pricing for details.
"""

raise ImportError(
    "nexus.providers.ensemble requires a Pro Nexus license. "
    "Visit https://gozerai.com/pricing for details."
)
PYEOF

mkdir -p "${TARGET}/src/nexus/providers/strategies"
cat > "${TARGET}/src/nexus/providers/strategies/__init__.py" << 'PYEOF'
"""Advanced ensemble strategies require a Pro license.

Visit https://gozerai.com/pricing for details.
"""

raise ImportError(
    "nexus.providers.strategies requires a Pro Nexus license. "
    "Visit https://gozerai.com/pricing for details."
)
PYEOF

echo ""
echo "=== Export complete: ${TARGET} ==="
echo ""
echo "Community-tier modules included:"
echo "  api, cli, core, memory, rag, storage, utils, agents,"
echo "  services, mcp_server, importers, orchestration, observatory,"
echo "  providers/adapters, providers/cost, providers/resilience,"
echo "  providers/safety, providers/monitoring, licensing"
echo ""
echo "Stripped (Pro/Enterprise/Private):"
echo "  reasoning, ensemble, strategies, discovery, intelligence,"
echo "  insights, cog_eng, experts, data, automations, blueprints,"
echo "  csuite, coo, gui, ui, helm, kubernetes"
