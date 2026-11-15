#!/usr/bin/env bash
# Start the interactive review (http://localhost:8000). Ctrl+C to stop.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/python ] || { echo "Run: bash scripts/setup_env.sh first"; exit 1; }
exec .venv/bin/python scripts/review.py "$@"
