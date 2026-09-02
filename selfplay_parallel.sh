#!/bin/bash
# Generate self-play games with several worker processes in parallel and
# concatenate them into one HCPE file.
#
# A single Python self-play process is CPU-bound on the MCTS tree and leaves the
# GPU mostly idle; running N workers that share the GPU multiplies throughput
# roughly N-fold (until the GPU or CPU cores saturate).
#
# Usage:
#   ./selfplay_parallel.sh <model.pth> <output.hcpe> [extra selfplay args...]
#
# Tunables (environment variables):
#   WORKERS    number of parallel processes        (default 8)
#   GAMES      total games across all workers       (default 1000)
#   PLAYOUTS   playouts per move                     (default 400)
#   BATCHSIZE  inference batch size per worker       (default 32)
#   GPU        GPU id (-1 for CPU)                   (default 0)
#   PYTHON     interpreter                           (default python)
#   METRICS_PREFIX  when set, worker w appends structured metrics to
#                   "<prefix>-w<w>.jsonl" (read by dashboard/app.py)
#   ITERATION       RL loop iteration number recorded in those metrics
set -e

MODEL="${1:?usage: selfplay_parallel.sh <model.pth> <output.hcpe> [args...]}"
OUT="${2:?usage: selfplay_parallel.sh <model.pth> <output.hcpe> [args...]}"
shift 2

WORKERS="${WORKERS:-8}"
GAMES="${GAMES:-1000}"
PLAYOUTS="${PLAYOUTS:-400}"
BATCHSIZE="${BATCHSIZE:-32}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-python}"
METRICS_PREFIX="${METRICS_PREFIX:-}"
ITERATION="${ITERATION:-}"

# 各ワーカーが受け持つ局数 (端数は切り上げ)
PER=$(( (GAMES + WORKERS - 1) / WORKERS ))
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

echo "spawning $WORKERS workers x $PER games (playouts=$PLAYOUTS batchsize=$BATCHSIZE gpu=$GPU)"
pids=()
for w in $(seq 0 $((WORKERS - 1))); do
    metrics_args=()
    if [ -n "$METRICS_PREFIX" ]; then
        metrics_args+=(--metrics "${METRICS_PREFIX}-w${w}.jsonl")
    fi
    if [ -n "$ITERATION" ]; then
        metrics_args+=(--iteration "$ITERATION")
    fi
    "$PYTHON" -m pydlshogi2.selfplay "$MODEL" "$TMPDIR/w$w.hcpe" \
        --games "$PER" --playouts "$PLAYOUTS" --batchsize "$BATCHSIZE" \
        --gpu "$GPU" --seed "$w" "${metrics_args[@]}" "$@" > "$TMPDIR/w$w.log" 2>&1 &
    pids+=($!)
done

# 全ワーカーの完了を待ち、いずれか失敗したら中断する
fail=0
for p in "${pids[@]}"; do
    wait "$p" || fail=1
done
if [ "$fail" != 0 ]; then
    echo "a self-play worker failed; logs in $TMPDIR (kept):"
    trap - EXIT
    tail -n 5 "$TMPDIR"/w*.log
    exit 1
fi

# HCPEは固定長レコードの連結なので cat でマージできる
# (自己対局では重複局面も価値学習の有効サンプルなので dedup しない)
cat "$TMPDIR"/w*.hcpe > "$OUT"
echo "merged $WORKERS files -> $OUT ($(stat -c%s "$OUT") bytes)"
