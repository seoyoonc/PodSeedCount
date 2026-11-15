#!/usr/bin/env bash
# Linux / macOS: create .venv and install the requirements.  Usage: bash scripts/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10 or newer is required"'
"$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
echo
echo "Done. Run:   .venv/bin/python scripts/run_count.py"
