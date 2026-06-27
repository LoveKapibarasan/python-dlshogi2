#!/bin/bash
# Launch a training job on a Vast.ai (or any CUDA Linux) instance.
#
# Bootstraps the virtual environment on first use (via vast_setup.sh), then runs
# the chosen task in the background with logging, so the job survives an SSH
# disconnect.
#
# Usage:
#   ./vast_run.sh train    train.hcpe test.hcpe --gpu 0 --amp
#   ./vast_run.sh selfplay checkpoints/checkpoint.pth selfplay.hcpe --games 1000 --gpu 0
#   ./vast_run.sh rl       checkpoints/checkpoint.pth            # full self-play loop
#
# Any extra arguments are forwarded verbatim to the underlying command. The RL
# loop is also tunable through its own environment variables (ITERATIONS,
# GAMES, PLAYOUTS, ...); see rl_loop.sh.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-}"
shift || true
PYTHON="$SCRIPT_DIR/.venv/bin/python"

# 初回はvenvを自動構築する
if [ ! -x "$PYTHON" ]; then
    echo "Virtual environment not found; running vast_setup.sh ..."
    "$SCRIPT_DIR/vast_setup.sh"
fi

mkdir -p logs
TS="$(date +%Y%m%d-%H%M%S)"
LOG="logs/${MODE:-none}-$TS.log"

case "$MODE" in
    train)
        nohup "$PYTHON" -m pydlshogi2.train "$@" > "$LOG" 2>&1 &
        ;;
    selfplay)
        nohup "$PYTHON" -m pydlshogi2.selfplay "$@" > "$LOG" 2>&1 &
        ;;
    rl)
        # rl_loop.sh shells out to python; point it at the venv interpreter.
        PYTHON="$PYTHON" nohup "$SCRIPT_DIR/rl_loop.sh" "$@" > "$LOG" 2>&1 &
        ;;
    *)
        echo "usage: $0 {train|selfplay|rl} [args...]"
        exit 1
        ;;
esac

PID=$!
echo "$PID" > "logs/${MODE}.pid"
echo "Started '$MODE' (pid $PID)."
echo "  log:    $LOG"
echo "  follow: tail -f $LOG"
echo "  stop:   kill \$(cat logs/${MODE}.pid)"
