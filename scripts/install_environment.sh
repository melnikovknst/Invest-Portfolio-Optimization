#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TASK_DIR=$(dirname "$PROJECT_DIR")
PYTHON_BIN="${PORTFOLIO_PYTHON:-/Users/konstantinmelnikov/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3}"
if [ ! -x "$PYTHON_BIN" ]; then PYTHON_BIN=python3; fi
"$PYTHON_BIN" -m venv "$TASK_DIR/.venv"
"$TASK_DIR/.venv/bin/python" -m pip install -r "$PROJECT_DIR/requirements.txt"
"$TASK_DIR/.venv/bin/python" -m pip freeze > "$PROJECT_DIR/requirements-lock.txt"
"$TASK_DIR/.venv/bin/python" -c 'import catboost, hmmlearn, sklearn, nbclient; print("Portfolio environment ready")'
