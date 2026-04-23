#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -x .venv/bin/python ]]; then
  PY=.venv/bin/python
elif [[ -x .venv/bin/python3 ]]; then
  PY=.venv/bin/python3
else
  PY=python3
fi
echo "Using: $PY"
"$PY" -m pytest tests/ -v --tb=short "$@"
