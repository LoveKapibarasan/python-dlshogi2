#!/bin/bash
# Train one human-imitation model per dan/kyu band, sequentially on one GPU.
#
# For human imitation the policy (move prediction) is the objective, so value is
# trained from the game outcome only (--val_lambda 1.0) and the metric to watch
# is the logged `test accuracy` (first number = policy move-match rate). The
# smaller bands overfit quickly, so keep the epoch count modest and pick the
# checkpoint with the lowest test loss for play (see human/README.md).
#
# Usage (run in the background so it survives an SSH disconnect):
#
#   cd ~/python-dlshogi2
#   nohup ./human/train_all_bands.sh > human_data/train_all.log 2>&1 &
#
# Tunables (environment variables):
#   BANDS      space-separated band dirs   (default: the nine single-rank bands
#              3級..六段 made by `--bands 28,29,30,31,32,33,34,35,36,37`)
#   DATA_DIR   dataset root                (default ~/human_data)
#   OUT_DIR    checkpoint output dir       (default = DATA_DIR)
#   EPOCHS     epochs per band             (default 6)
#   BATCHSIZE  training batch size         (default 256)
#   LR         learning rate               (default 0.01)
#   GPU        GPU id                      (default 0)
#   BLOCKS     residual blocks             (default 10)
#   CHANNELS   channel width               (default 192)
#   AMP_DTYPE  autocast dtype              (default float16; bfloat16 on Ampere+)
#   INIT_MODEL fine-tune every band from this base checkpoint (default: from scratch)
#   TRAIN_NAME train file name inside each band dir (default train.hcpe)
#   PYTHON     interpreter                 (default ../.venv/bin/python)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BANDS="${BANDS:-0028-0028 0029-0029 0030-0030 0031-0031 0032-0032 0033-0033 0034-0034 0035-0035 0036-0036}"
DATA_DIR="${DATA_DIR:-$HOME/human_data}"
OUT_DIR="${OUT_DIR:-$DATA_DIR}"
EPOCHS="${EPOCHS:-6}"
BATCHSIZE="${BATCHSIZE:-256}"
LR="${LR:-0.01}"
GPU="${GPU:-0}"
BLOCKS="${BLOCKS:-10}"
CHANNELS="${CHANNELS:-192}"
AMP_DTYPE="${AMP_DTYPE:-float16}"
INIT_MODEL="${INIT_MODEL:-}"
TRAIN_NAME="${TRAIN_NAME:-train.hcpe}"
PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python}"

cd "$REPO_DIR"
for band in $BANDS; do
    train="$DATA_DIR/$band/$TRAIN_NAME"
    test="$DATA_DIR/$band/test.hcpe"
    if [ ! -s "$train" ] || [ "$(stat -L -c %s "$test" 2>/dev/null || echo 0)" -lt $((1024 * 38)) ]; then
        echo "=== skip $band (missing $train, or $test has < 1024 positions) ==="
        continue
    fi
    echo "=== training band: $band ($(date '+%F %T')) ==="
    init_args=()
    [ -n "$INIT_MODEL" ] && init_args=(--init_model "$INIT_MODEL")
    "$PYTHON" -m pydlshogi2.train "$train" "$test" \
        --gpu "$GPU" --amp --amp_dtype "$AMP_DTYPE" \
        --blocks "$BLOCKS" --channels "$CHANNELS" --epoch "$EPOCHS" --batchsize "$BATCHSIZE" --lr "$LR" \
        "${init_args[@]}" --val_lambda 1.0 --save_interval 2000 \
        --log "$OUT_DIR/train-$band.log" \
        --checkpoint "$OUT_DIR/model-$band-{epoch:03}.pth"
    echo "=== done $band ($(date '+%F %T')) ==="
done
echo "all bands finished"
