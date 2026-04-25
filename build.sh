#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

echo "[1/3] Creating virtual environment..."
python3 -m venv "$VENV"

echo "[2/3] Installing dependencies..."
"$VENV/bin/pip" install --upgrade pip --quiet
"$VENV/bin/pip" install -r "$SCRIPT_DIR/requirements.txt" --quiet

echo "[3/3] Installing package..."
"$VENV/bin/pip" install -e "$SCRIPT_DIR" --quiet

echo ""
echo "Build complete. Run the engine with:"
echo "  ./mcts_player.sh"
echo "  ./onnx_player.sh"
