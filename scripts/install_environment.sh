#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON_BIN="${PORTFOLIO_PYTHON:-python3}"
ENVIRONMENT_DIR="$PROJECT_DIR/.venv"
if [ -L "$ENVIRONMENT_DIR" ]; then
  echo "Refusing to modify a linked environment; create a repository-local .venv first." >&2
  exit 1
fi
"$PYTHON_BIN" -m venv "$ENVIRONMENT_DIR"
"$ENVIRONMENT_DIR/bin/python" -m pip install -r "$PROJECT_DIR/requirements.txt"
# requirements-lock.txt is historical run evidence, not the new machine's lock.
"$ENVIRONMENT_DIR/bin/python" -c 'import catboost, hmmlearn, sklearn, nbclient; print("Portfolio environment ready")'
