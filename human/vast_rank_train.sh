#!/usr/bin/env bash
# Train the Shogi Wars rank models on a Vast.ai CUDA instance.
#
# Copy a prebuilt DATA_DIR to the instance first, then run this from the repo:
#
#   DATA_DIR=/workspace/human_data_ranks/20260926 ./human/vast_rank_train.sh
#
# By default this trains all nine rank bands (3級 .. 六段), using the existing
# HCPE files and the shared base-model/fine-tuning pipeline. Set BANDS to a
# space-separated subset, e.g. "0029-0029 0031-0031 0033-0033", to train only
# 2級, 初段 and 三段. The job backgrounds itself and writes a timestamped log
# under DATA_DIR/logs so an SSH disconnect does not stop training.
set -euo pipefail

SCRIPT_PATH="$(realpath "$0")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${DATA_DIR:-$HOME/human_data_ranks/latest}"
BUNDLE="${DATA_BUNDLE:-}"
BANDS="${BANDS:-0028-0028 0029-0029 0030-0030 0031-0031 0032-0032 0033-0033 0034-0034 0035-0035 0036-0036}"
PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python}"
MODEL_DIR="${MODEL_DIR:-$DATA_DIR/onnx}"

mkdir -p "$DATA_DIR/logs"

if [ "${VAST_RANK_TRAIN_FOREGROUND:-0}" != "1" ]; then
    LOG="$DATA_DIR/logs/vast-rank-$(date '+%Y%m%d-%H%M%S').log"
    nohup env VAST_RANK_TRAIN_FOREGROUND=1 "$SCRIPT_PATH" > "$LOG" 2>&1 < /dev/null &
    PID=$!
    echo "$PID" > "$DATA_DIR/logs/vast-rank.pid"
    echo "Started rank-model training (pid $PID)."
    echo "Log: $LOG"
    exit 0
fi

cd "$REPO_DIR"

if [ -n "$BUNDLE" ]; then
    [ -s "$BUNDLE" ] || { echo "missing dataset bundle $BUNDLE" >&2; exit 1; }
    tar -xzf "$BUNDLE" -C "$DATA_DIR"
fi

if [ ! -x "$PYTHON" ]; then
    echo "Creating the training environment with vast_setup.sh ..."
    "$REPO_DIR/vast_setup.sh"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found; run this on a CUDA-enabled Vast.ai instance." >&2
    exit 1
fi
nvidia-smi
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA is not available"; x=torch.nn.Conv2d(3, 8, 3).cuda()(torch.randn(1, 3, 16, 16, device="cuda")); torch.cuda.synchronize(); print("torch", torch.__version__, "CUDA", torch.version.cuda, "capability", torch.cuda.get_device_capability(), "conv", tuple(x.shape))'

for band in $BANDS; do
    [ -s "$DATA_DIR/$band/train.hcpe" ] || { echo "missing $DATA_DIR/$band/train.hcpe" >&2; exit 1; }
    [ -s "$DATA_DIR/$band/test.hcpe" ] || { echo "missing $DATA_DIR/$band/test.hcpe" >&2; exit 1; }
done
[ -s "$DATA_DIR/DATASET.md" ] || { echo "missing $DATA_DIR/DATASET.md; copy the dataset manifest too" >&2; exit 1; }

export DATA_DIR BANDS PYTHON
export SKIP_DATA_BUILD="${SKIP_DATA_BUILD:-1}"
export GPU="${GPU:-0}"
export EPOCHS="${EPOCHS:-2}"
export BASE_EPOCHS="${BASE_EPOCHS:-1}"
export BASE_POSITIONS="${BASE_POSITIONS:-20000000}"
export BAND_POSITIONS="${BAND_POSITIONS:-4000000}"
export BATCHSIZE="${BATCHSIZE:-256}"
export BLOCKS="${BLOCKS:-10}"
export CHANNELS="${CHANNELS:-192}"
export AMP_DTYPE="${AMP_DTYPE:-float16}"
export FT_LR="${FT_LR:-0.005}"

echo "Training rank bands: $BANDS"
echo "Dataset: $DATA_DIR"
echo "GPU: $GPU; epochs per band: $EPOCHS; per-band position cap: $BAND_POSITIONS"
"$SCRIPT_DIR/run_rank_pipeline.sh"

mkdir -p "$MODEL_DIR"
: > "$DATA_DIR/eval-policy.tsv"
for band in $BANDS; do
    checkpoint="$DATA_DIR/model-$band-$(printf '%03d' "$EPOCHS").pth"
    [ -s "$checkpoint" ] || { echo "missing trained checkpoint $checkpoint" >&2; exit 1; }
    "$PYTHON" human/eval_policy.py \
        --models "$checkpoint" \
        --tests "$DATA_DIR/$band/test.hcpe" \
        --max_positions 50000 \
        --batchsize "${EVAL_BATCHSIZE:-256}" \
        --gpu "$GPU" \
        --amp_dtype "$AMP_DTYPE" >> "$DATA_DIR/eval-policy.tsv"
    "$PYTHON" utils/export_onnx.py "$checkpoint" "$MODEL_DIR/rank-$band.onnx"
done
echo "Rank models exported to $MODEL_DIR"
echo "Policy match results: $DATA_DIR/eval-policy.tsv"
