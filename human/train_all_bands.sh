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
#   BANDS      space-separated band dirs   (default "kyu dan1-3 dan7plus")
#   DATA_DIR   dataset root                (default ~/human_data)
#   OUT_DIR    checkpoint output dir       (default = DATA_DIR)
#   EPOCHS     epochs per band             (default 6)
#   BATCHSIZE  training batch size         (default 256)
#   LR         learning rate               (default 0.01)
#   GPU        GPU id                      (default 0)
#   PYTHON     interpreter                 (default ../.venv/bin/python)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BANDS="${BANDS:-kyu dan1-3 dan7plus}"
DATA_DIR="${DATA_DIR:-$HOME/human_data}"
OUT_DIR="${OUT_DIR:-$DATA_DIR}"
EPOCHS="${EPOCHS:-6}"
BATCHSIZE="${BATCHSIZE:-256}"
LR="${LR:-0.01}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python}"

cd "$REPO_DIR"
for band in $BANDS; do
    train="$DATA_DIR/$band/train.hcpe"
    test="$DATA_DIR/$band/test.hcpe"
    if [ ! -s "$train" ] || [ ! -s "$test" ]; then
        echo "=== skip $band (missing $train / $test) ==="
        continue
    fi
    echo "=== training band: $band ($(date '+%F %T')) ==="
    "$PYTHON" -m pydlshogi2.train "$train" "$test" \
        --gpu "$GPU" --amp --epoch "$EPOCHS" --batchsize "$BATCHSIZE" --lr "$LR" \
        --val_lambda 1.0 --save_interval 2000 \
        --log "$OUT_DIR/train-$band.log" \
        --checkpoint "$OUT_DIR/model-$band-{epoch:03}.pth"
    echo "=== done $band ($(date '+%F %T')) ==="
done
echo "all bands finished"
