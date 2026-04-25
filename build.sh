#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

echo "[1/4] Creating virtual environment..."
python3 -m venv "$VENV"

echo "[2/4] Installing dependencies..."
"$VENV/bin/pip" install --upgrade pip --quiet
"$VENV/bin/pip" install -r "$SCRIPT_DIR/requirements.txt" --quiet
"$VENV/bin/pip" install -e "$SCRIPT_DIR" --quiet
"$VENV/bin/pip" install pyinstaller --quiet

HIDDEN=(
    --hidden-import pydlshogi2
    --hidden-import pydlshogi2.features
    --hidden-import pydlshogi2.network.policy_value_resnet
    --hidden-import pydlshogi2.uct.uct_node
    --hidden-import pydlshogi2.player.base_player
    --hidden-import pydlshogi2.player.mcts_player
    --hidden-import pydlshogi2.player.onnx_player
)

echo "[3/4] Building mcts_player..."
"$VENV/bin/pyinstaller" \
    --name mcts_player \
    --onedir \
    --collect-all torch \
    --collect-all numpy \
    --collect-all cshogi \
    "${HIDDEN[@]}" \
    "$SCRIPT_DIR/pydlshogi2/player/mcts_player.py"

echo "[4/4] Building onnx_player..."
"$VENV/bin/pyinstaller" \
    --name onnx_player \
    --onedir \
    --collect-all torch \
    --collect-all numpy \
    --collect-all onnxruntime \
    --collect-all cshogi \
    "${HIDDEN[@]}" \
    "$SCRIPT_DIR/pydlshogi2/player/onnx_player.py"

echo ""
echo "Done. No Python needed to run:"
echo "  ./dist/mcts_player/mcts_player"
echo "  ./dist/onnx_player/onnx_player"
