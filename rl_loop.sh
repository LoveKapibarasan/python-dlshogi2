#!/bin/bash
# Reinforcement-learning self-play loop.
#
# Each iteration:
#   1. generates self-play games with the current model,
#   2. fine-tunes the model on the freshly generated data (resuming from the
#      current checkpoint),
#   3. promotes the new checkpoint to "current" for the next iteration.
#
# Start from a supervised-pretrained checkpoint for a warm start. Usage:
#
#   ./rl_loop.sh checkpoints/checkpoint.pth
#
# Tune the loop with the environment variables below.
set -e

INIT_MODEL="${1:?usage: rl_loop.sh <initial_checkpoint.pth>}"

ITERATIONS="${ITERATIONS:-20}"     # number of self-play/train cycles
GAMES="${GAMES:-1000}"             # self-play games per iteration (across all workers)
PLAYOUTS="${PLAYOUTS:-400}"        # MCTS playouts per move
WORKERS="${WORKERS:-8}"            # parallel self-play workers (saturate the GPU)
SELFPLAY_BATCHSIZE="${SELFPLAY_BATCHSIZE:-32}"  # inference batch size per worker
EPOCHS="${EPOCHS:-1}"              # training epochs per iteration
BATCHSIZE="${BATCHSIZE:-1024}"     # training batch size
LR="${LR:-0.002}"                  # lower LR than SL: fine-tuning
VAL_LAMBDA="${VAL_LAMBDA:-0.5}"    # blend game result with bootstrapped value
GPU="${GPU:-0}"
WORKDIR="${WORKDIR:-rl}"           # where iteration artifacts are written
PYTHON="${PYTHON:-python}"         # python interpreter (set to .venv/bin/python on Vast.ai)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$WORKDIR"
CURRENT="$INIT_MODEL"

for i in $(seq 1 "$ITERATIONS"); do
    echo "=== RL iteration $i / $ITERATIONS (model: $CURRENT) ==="
    DATA="$WORKDIR/selfplay-$(printf '%03d' "$i").hcpe"
    NEXT="$WORKDIR/checkpoint-$(printf '%03d' "$i").pth"

    # 既に学習済みのイテレーションはスキップ (クラッシュ/preemptionからの再開)
    if [ -s "$NEXT" ]; then
        echo "iteration $i already trained ($NEXT); skipping"
        CURRENT="$NEXT"
        continue
    fi

    # 生成済みの自己対局データがあれば再利用する
    if [ -s "$DATA" ]; then
        echo "[1/2] reusing existing self-play data -> $DATA"
    else
        echo "[1/2] parallel self-play ($WORKERS workers) -> $DATA"
        WORKERS="$WORKERS" GAMES="$GAMES" PLAYOUTS="$PLAYOUTS" \
            BATCHSIZE="$SELFPLAY_BATCHSIZE" GPU="$GPU" PYTHON="$PYTHON" \
            "$SCRIPT_DIR/selfplay_parallel.sh" "$CURRENT" "$DATA"
    fi

    echo "[2/2] train -> $NEXT"
    # Train on all self-play data generated so far (the test split reuses the
    # latest batch for a quick sanity metric).
    "$PYTHON" -m pydlshogi2.train "$WORKDIR"/selfplay-*.hcpe "$DATA" \
        --resume "$CURRENT" \
        --epoch "$EPOCHS" \
        --batchsize "$BATCHSIZE" \
        --lr "$LR" \
        --val_lambda "$VAL_LAMBDA" \
        --gpu "$GPU" \
        --checkpoint "$NEXT"

    CURRENT="$NEXT"
done

echo "RL loop finished. Final model: $CURRENT"
