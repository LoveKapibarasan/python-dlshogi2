#!/bin/bash
# Provision a Vast.ai (or any CUDA Linux) instance for training.
#
# Creates a virtual environment, installs the CUDA build of PyTorch and the
# rest of the dependencies, swaps the CPU onnxruntime for the GPU build, and
# installs this package in editable mode.
#
# Run once from the repository root:
#
#   ./vast_setup.sh
#
# Override the interpreter with PYBIN=python3.12 ./vast_setup.sh if needed.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYBIN="${PYBIN:-python3}"
VENV="$SCRIPT_DIR/.venv"

echo "[1/4] Creating virtual environment with $PYBIN ..."
"$PYBIN" -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip --quiet

echo "[2/4] Installing dependencies (CUDA torch from PyPI) ..."
# The default PyPI torch wheel is CUDA-enabled on Linux x86_64.
"$VENV/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"

echo "[3/4] Swapping onnxruntime -> onnxruntime-gpu ..."
"$VENV/bin/pip" uninstall -y onnxruntime >/dev/null 2>&1 || true
"$VENV/bin/pip" install onnxruntime-gpu || echo "  (onnxruntime-gpu unavailable, keeping CPU build)"

echo "[4/4] Installing package (editable) ..."
"$VENV/bin/pip" install -e "$SCRIPT_DIR"

echo ""
echo "Environment ready. CUDA check:"
"$VENV/bin/python" -c "import torch; print('  torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
echo ""
echo "Next: ./vast_run.sh train  ...   (or selfplay / rl). See README."
