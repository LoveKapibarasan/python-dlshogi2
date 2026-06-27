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
GAMES="${GAMES:-1000}"             # self-play games per iteration
PLAYOUTS="${PLAYOUTS:-800}"        # MCTS playouts per move
EPOCHS="${EPOCHS:-1}"              # training epochs per iteration
BATCHSIZE="${BATCHSIZE:-1024}"
LR="${LR:-0.002}"                  # lower LR than SL: fine-tuning
VAL_LAMBDA="${VAL_LAMBDA:-0.5}"    # blend game result with bootstrapped value
GPU="${GPU:-0}"
WORKDIR="${WORKDIR:-rl}"           # where iteration artifacts are written
PYTHON="${PYTHON:-python}"         # python interpreter (set to .venv/bin/python on Vast.ai)

mkdir -p "$WORKDIR"
CURRENT="$INIT_MODEL"

for i in $(seq 1 "$ITERATIONS"); do
    echo "=== RL iteration $i / $ITERATIONS (model: $CURRENT) ==="
    DATA="$WORKDIR/selfplay-$(printf '%03d' "$i").hcpe"
    NEXT="$WORKDIR/checkpoint-$(printf '%03d' "$i").pth"

    echo "[1/2] self-play -> $DATA"
    "$PYTHON" -m pydlshogi2.selfplay "$CURRENT" "$DATA" \
        --games "$GAMES" --playouts "$PLAYOUTS" --gpu "$GPU"

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
