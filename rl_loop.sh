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
METRICS_DIR="${METRICS_DIR:-$WORKDIR/metrics}"  # structured metrics for the dashboard
WINDOW="${WINDOW:-0}"              # train on the latest N iterations' data only (0 = all)
SELFPLAY_ARGS="${SELFPLAY_ARGS:-}" # extra pydlshogi2.selfplay arguments (e.g. --temp_cutoff 999)
TRAIN_ARGS="${TRAIN_ARGS:-}"       # extra pydlshogi2.train arguments (e.g. --amp --amp_dtype float16)
PRUNE_CHECKPOINTS="${PRUNE_CHECKPOINTS:-}"  # set to 1 to keep only every 10th old checkpoint
# 昇格ゲート (EXP-003): 新しいチェックポイントを現行の最良と対局させ、
# 勝率が GATE_THRESHOLD 以上のときだけ次のイテレーションの自己対局に使う
GATE_GAMES="${GATE_GAMES:-0}"       # 0 = gate off (every checkpoint is promoted)
GATE_PLAYOUTS="${GATE_PLAYOUTS:-400}"
GATE_THRESHOLD="${GATE_THRESHOLD:-0.5}"
# GATE_ENGINE / GATE_OPENING の既定値は SCRIPT_DIR が決まってから (下で) 設定する

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GATE_ENGINE="${GATE_ENGINE:-$SCRIPT_DIR/usi_engine_native.sh}"
GATE_OPENING="${GATE_OPENING:-$SCRIPT_DIR/openings.txt}"

mkdir -p "$WORKDIR" "$METRICS_DIR"
CURRENT="$INIT_MODEL"

# ループ全体を1つのrunとして記録する (各イテレーションはこのrun配下のイベント)
RL_LOG="$METRICS_DIR/rl.jsonl"
RL_RUN_ID="${RL_RUN_ID:-rl-$(date +%Y%m%d-%H%M%S)}"
"$PYTHON" -m pydlshogi2.metrics "$RL_LOG" --type run --kind rl --run-id "$RL_RUN_ID" \
    --set init_model="$INIT_MODEL" --set iterations="$ITERATIONS" \
    --set games="$GAMES" --set playouts="$PLAYOUTS" --set workers="$WORKERS" \
    --set epochs="$EPOCHS" --set batchsize="$BATCHSIZE" --set lr="$LR" \
    --set val_lambda="$VAL_LAMBDA" --set workdir="$WORKDIR" > /dev/null

for i in $(seq 1 "$ITERATIONS"); do
    echo "=== RL iteration $i / $ITERATIONS (model: $CURRENT) ==="
    DATA="$WORKDIR/selfplay-$(printf '%03d' "$i").hcpe"
    NEXT="$WORKDIR/checkpoint-$(printf '%03d' "$i").pth"

    GATE_FILE="$WORKDIR/gate-$(printf '%03d' "$i").txt"
    # 既に学習済みのイテレーションはスキップ (クラッシュ/preemptionからの再開)
    if [ -s "$NEXT" ] || [ -s "$GATE_FILE" ]; then
        echo "iteration $i already trained ($NEXT); skipping"
        # ゲートで落ちたチェックポイントは最良にしない
        if ! grep -q '^reject' "$GATE_FILE" 2>/dev/null; then
            CURRENT="$NEXT"
        fi
        continue
    fi

    ITER_STARTED=$(date +%s)
    "$PYTHON" -m pydlshogi2.metrics "$RL_LOG" --type event --event iteration_start \
        --run-id "$RL_RUN_ID" --set iteration="$i" --set model="$CURRENT" \
        --set data="$DATA" > /dev/null

    # 生成済みの自己対局データがあれば再利用する
    if [ -s "$DATA" ]; then
        echo "[1/2] reusing existing self-play data -> $DATA"
    else
        echo "[1/2] parallel self-play ($WORKERS workers) -> $DATA"
        WORKERS="$WORKERS" GAMES="$GAMES" PLAYOUTS="$PLAYOUTS" \
            BATCHSIZE="$SELFPLAY_BATCHSIZE" GPU="$GPU" PYTHON="$PYTHON" \
            METRICS_PREFIX="$METRICS_DIR/selfplay-$(printf '%03d' "$i")" ITERATION="$i" \
            "$SCRIPT_DIR/selfplay_parallel.sh" "$CURRENT" "$DATA" $SELFPLAY_ARGS
    fi

    echo "[2/2] train -> $NEXT"
    # Train on the self-play data of the latest WINDOW iterations (all of it when
    # WINDOW=0).  From a random start the oldest games are the weakest, and
    # keeping them forever drags the model back towards random play.  The test
    # split reuses the latest batch for a quick sanity metric.
    TRAIN_DATA=( "$WORKDIR"/selfplay-*.hcpe )
    if [ "$WINDOW" -gt 0 ] && [ "${#TRAIN_DATA[@]}" -gt "$WINDOW" ]; then
        TRAIN_DATA=( "${TRAIN_DATA[@]: -$WINDOW}" )
    fi
    "$PYTHON" -m pydlshogi2.train "${TRAIN_DATA[@]}" "$DATA" $TRAIN_ARGS \
        --resume "$CURRENT" \
        --epoch "$EPOCHS" \
        --batchsize "$BATCHSIZE" \
        --lr "$LR" \
        --val_lambda "$VAL_LAMBDA" \
        --gpu "$GPU" \
        --checkpoint "$NEXT" \
        --metrics "$METRICS_DIR/train-$(printf '%03d' "$i").jsonl" \
        --run_id "$RL_RUN_ID-train-$(printf '%03d' "$i")"

    "$PYTHON" -m pydlshogi2.metrics "$RL_LOG" --type metric \
        --run-id "$RL_RUN_ID" --set scope=iteration --set iteration="$i" \
        --set model="$CURRENT" --set checkpoint="$NEXT" --set data="$DATA" \
        --set data_bytes="$(stat -c%s "$DATA")" \
        --set seconds="$(( $(date +%s) - ITER_STARTED ))" > /dev/null

    PROMOTE=1
    if [ "$GATE_GAMES" -gt 0 ]; then
        echo "[gate] $NEXT vs $CURRENT ($GATE_GAMES games, $GATE_PLAYOUTS playouts)"
        GATE_OUT="$("$PYTHON" -m pydlshogi2.match \
            --engine1 "$GATE_ENGINE" --name1 "rl-$(printf '%03d' "$i")" \
            --engine2 "$GATE_ENGINE" --name2 "best" \
            --options1 "modelfile=$NEXT" --options2 "modelfile=$CURRENT" \
            --games "$GATE_GAMES" --playouts "$GATE_PLAYOUTS" --opening "$GATE_OPENING" \
            --metrics "$METRICS_DIR/gate-$(printf '%03d' "$i").jsonl" \
            --experiment "rl-gate" --quiet 2>&1)" || true
        SCORE="$(echo "$GATE_OUT" | awk '/^score/ {print $3}')"
        if [ -z "$SCORE" ] || ! awk -v s="$SCORE" -v t="$GATE_THRESHOLD" 'BEGIN { exit !(s >= t) }'; then
            PROMOTE=0
        fi
        echo "$([ $PROMOTE = 1 ] && echo accept || echo reject) score=$SCORE vs=$CURRENT" > "$GATE_FILE"
        echo "[gate] $(cat "$GATE_FILE")"
        echo "$GATE_OUT" | grep -E "^(W-L-D|Elo)" || true
        # 対局が成立しなかったときは原因を残す
        [ -z "$SCORE" ] && echo "$GATE_OUT" | tail -20 >> "$GATE_FILE"
    fi

    # 古いチェックポイントは容量を食うので、10 イテレーションごとのものだけ残す (最良は残す)
    if [ -n "$PRUNE_CHECKPOINTS" ] && [ "$i" -gt 2 ]; then
        OLD="$WORKDIR/checkpoint-$(printf '%03d' "$((i - 2))").pth"
        if [ $(((i - 2) % 10)) -ne 0 ] && [ "$OLD" != "$CURRENT" ]; then
            rm -f "$OLD"
        fi
    fi

    if [ "$PROMOTE" = 1 ]; then
        CURRENT="$NEXT"
    else
        # 昇格しなかったチェックポイントは使わないので消す
        [ -n "$PRUNE_CHECKPOINTS" ] && rm -f "$NEXT"
    fi
    echo "$CURRENT" > "$WORKDIR/best.txt"
done

"$PYTHON" -m pydlshogi2.metrics "$RL_LOG" --type event --event run_end \
    --run-id "$RL_RUN_ID" --set status=completed --set final_model="$CURRENT" > /dev/null

echo "RL loop finished. Final model: $CURRENT"
