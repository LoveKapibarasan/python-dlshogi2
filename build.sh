#!/bin/bash
# Build standalone executables for mcts_player and onnx_player using PyInstaller.
# Run this script once on each target platform (Linux → Linux binary, Windows → .exe).
# The resulting binaries do NOT require Python or any packages to be installed.
#
# Output:
#   dist/mcts_player/mcts_player[.exe]   – PyTorch backend
#   dist/onnx_player/onnx_player[.exe]   – ONNX backend (lighter at runtime)
#
# Usage:
#   chmod +x build.sh
#   ./build.sh
#
# To use the built engine in a shogi GUI, point it at the executable inside dist/.
# Set the `modelfile` USI option to your .pth or .onnx file path.

set -e

# ── 1. Install build tool ────────────────────────────────────────────────────
echo "[1/3] Installing PyInstaller..."
pip install pyinstaller --quiet

# ── 2. Shared hidden imports (subpackages PyInstaller won't auto-detect) ─────
HIDDEN=(
    --hidden-import pydlshogi2
    --hidden-import pydlshogi2.features
    --hidden-import pydlshogi2.network.policy_value_resnet
    --hidden-import pydlshogi2.uct.uct_node
    --hidden-import pydlshogi2.player.base_player
    --hidden-import pydlshogi2.player.mcts_player
    --hidden-import pydlshogi2.player.onnx_player
)

# ── 3a. mcts_player (PyTorch backend, ~1–2 GB) ───────────────────────────────
echo "[2/3] Building mcts_player..."
pyinstaller \
    --name mcts_player \
    --onedir \
    --collect-all torch \
    --collect-all cshogi \
    "${HIDDEN[@]}" \
    pydlshogi2/player/mcts_player.py

# ── 3b. onnx_player (ONNX backend, ~300–500 MB) ──────────────────────────────
echo "[3/3] Building onnx_player..."
pyinstaller \
    --name onnx_player \
    --onedir \
    --collect-all torch \
    --collect-all onnxruntime \
    --collect-all cshogi \
    "${HIDDEN[@]}" \
    pydlshogi2/player/onnx_player.py

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "Build complete."
echo ""
echo "  dist/mcts_player/mcts_player   ← register this in your shogi GUI"
echo "  dist/onnx_player/onnx_player   ← lighter option (needs .onnx model)"
echo ""
echo "The model file is NOT bundled. Set the 'modelfile' USI option to:"
echo "  checkpoints/checkpoint.pth     (for mcts_player)"
echo "  model/model-0000225kai.onnx    (for onnx_player)"
