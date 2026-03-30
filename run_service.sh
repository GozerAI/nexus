#!/usr/bin/env bash
# Start Nexus as shared infrastructure + C-Suite bridge.
# Requires: Redis on localhost:6379 (already running via csuite-redis container)
#
# Usage:
#   ./run_service.sh              # default (supervised mode)
#   ./run_service.sh autonomous   # autonomous COO mode

set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-supervised}"

# Load API keys from .env (but override runtime flags below)
set -a
source .env
set +a

# Runtime overrides — don't use test/mock settings
export TESTING=False
export MOCK_LLM_RESPONSES=False
export BYPASS_AUTH_FOR_TESTS=False

# Nexus service config
export NEXUS_ENABLE_LEGACY_COO=true
export COO_MODE="$MODE"
export REDIS_URL="redis://:${REDIS_PASSWORD:-csuite_dev_redis}@${REDIS_HOST:-localhost}:${REDIS_PORT:-6379}"
export CHANNEL_PREFIX="csuite:nexus"
export HEALTH_PORT=8080
export LOG_LEVEL=INFO

echo "Starting Nexus shared infrastructure service"
echo "  COO mode:    $MODE"
echo "  Redis:       $REDIS_URL"
echo "  Channels:    $CHANNEL_PREFIX"
echo "  Health:      http://localhost:$HEALTH_PORT/health"
echo ""

python -m nexus.service
