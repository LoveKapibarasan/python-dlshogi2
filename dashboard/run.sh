#!/bin/bash
# Run the metrics dashboard, in the foreground or as a background service.
#
# Bootstraps its own virtual environment on first use, so the dashboard's
# dependencies (streamlit, pandas, altair) never land in the environment used
# for training or playing.
#
# Usage:
#   ./dashboard/run.sh              # foreground (Ctrl-C to stop)
#   ./dashboard/run.sh start        # background; survives an SSH disconnect
#   ./dashboard/run.sh stop
#   ./dashboard/run.sh restart
#   ./dashboard/run.sh status
#
# Tunables (environment variables):
#   PORT            listen port                      (default 8501)
#   ADDRESS         listen address                   (default 127.0.0.1)
#   METRICS_DIR     metrics directory                (default metrics)
#   CHECKPOINT_DIR  checkpoint directory             (default checkpoints)
#   VENV            virtual environment path         (default dashboard/.venv)
#   PYTHON          interpreter used to build VENV   (default python3)
#
# The default address is 127.0.0.1: the dashboard has no authentication, so it
# is not exposed beyond the machine it runs on. To view a remote one, forward
# the port instead:
#
#   ssh -L 8501:127.0.0.1:8501 <host>
#
# Set ADDRESS=0.0.0.0 only on a network you control.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PORT="${PORT:-8501}"
ADDRESS="${ADDRESS:-127.0.0.1}"
METRICS_DIR="${METRICS_DIR:-metrics}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"
VENV="${VENV:-$SCRIPT_DIR/.venv}"
PYTHON="${PYTHON:-python3}"

PID_FILE="$REPO_ROOT/logs/dashboard.pid"
LOG_FILE="$REPO_ROOT/logs/dashboard.log"

# 依存が揃った専用のvenvを用意する (エンジン側の環境は汚さない)
ensure_venv() {
    if [ ! -x "$VENV/bin/streamlit" ]; then
        echo "Setting up the dashboard environment in $VENV ..."
        "$PYTHON" -m venv "$VENV"
        "$VENV/bin/pip" install --upgrade pip --quiet
        "$VENV/bin/pip" install --quiet -r "$SCRIPT_DIR/requirements.txt"
    fi
}

running_pid() {
    [ -f "$PID_FILE" ] || return 1
    local pid
    pid="$(cat "$PID_FILE")"
    # PIDファイルが残っていてもプロセスが死んでいることがある
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "$pid"
        return 0
    fi
    return 1
}

streamlit_args() {
    echo "run $SCRIPT_DIR/app.py
--server.port=$PORT
--server.address=$ADDRESS
--server.headless=true
--browser.gatherUsageStats=false"
}

start_foreground() {
    ensure_venv
    echo "dashboard on http://$ADDRESS:$PORT  (metrics: $METRICS_DIR)"
    DLSHOGI_METRICS_DIR="$METRICS_DIR" DLSHOGI_CHECKPOINT_DIR="$CHECKPOINT_DIR" \
        exec "$VENV/bin/streamlit" $(streamlit_args)
}

start_background() {
    if pid="$(running_pid)"; then
        echo "already running (pid $pid) on http://$ADDRESS:$PORT"
        return 0
    fi
    ensure_venv
    mkdir -p "$REPO_ROOT/logs"
    DLSHOGI_METRICS_DIR="$METRICS_DIR" DLSHOGI_CHECKPOINT_DIR="$CHECKPOINT_DIR" \
        nohup "$VENV/bin/streamlit" $(streamlit_args) >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "started (pid $(cat "$PID_FILE")) on http://$ADDRESS:$PORT"
    echo "  log:  tail -f $LOG_FILE"
    echo "  stop: $0 stop"
}

stop_service() {
    if ! pid="$(running_pid)"; then
        echo "not running"
        rm -f "$PID_FILE"
        return 0
    fi
    kill "$pid"
    # 落ちるまで少し待ち、居座るようなら強制終了する
    for _ in $(seq 1 20); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.5
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "did not exit; sending SIGKILL"
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
    echo "stopped"
}

case "${1:-}" in
    ''|run|foreground)
        start_foreground
        ;;
    start)
        start_background
        ;;
    stop)
        stop_service
        ;;
    restart)
        stop_service
        start_background
        ;;
    status)
        if pid="$(running_pid)"; then
            echo "running (pid $pid) on http://$ADDRESS:$PORT"
        else
            echo "not running"
            exit 1
        fi
        ;;
    *)
        echo "usage: $0 [run|start|stop|restart|status]" >&2
        exit 1
        ;;
esac
