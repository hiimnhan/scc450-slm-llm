#!/usr/bin/env bash
# Quick bootstrap script: install uv, create/activate the venv, install deps + extras.
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v uv >/dev/null 2>&1; then
  command -v curl >/dev/null 2>&1 || { echo "curl is required to install uv." >&2; exit 1; }
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  if [[ -d "$HOME/.local/bin" && ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi

if [[ ! -d ".venv" ]]; then
  echo "Creating virtual environment..."
  uv venv
else
  echo "Virtual environment already exists."
fi

case "$(uname -s)" in
  Linux|Darwin) ACTIVATE=".venv/bin/activate" ;;
  MINGW*|MSYS*|CYGWIN*|Windows_NT) ACTIVATE=".venv/Scripts/activate" ;;
  *) echo "Unknown OS $(uname -s), defaulting to POSIX layout." >&2; ACTIVATE=".venv/bin/activate" ;;
esac

[[ -f "$ACTIVATE" ]] || { echo "Cannot find $ACTIVATE to activate the environment." >&2; exit 1; }
# shellcheck disable=SC1091
source "$ACTIVATE"

echo "Installing project dependencies..."
uv sync
echo "Installing extra packages (torch, docling)..."
uv pip install torch docling

cat <<'EOF'

Project environment ready.

Activate it manually when needed:
  source .venv/bin/activate   # Linux/macOS
  .\.venv\Scripts\activate    # Windows PowerShell
EOF
