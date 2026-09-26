#!/bin/sh
# USI engine launcher for the native-search player (pydlshogi2/uct/native).
# Same as usi_engine.sh, but runs NativeMCTSPlayer; build the library first with
# pydlshogi2/uct/native/build.sh.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -z "$PYTHON" ]; then
    if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
        PYTHON="$SCRIPT_DIR/.venv/bin/python"
    else
        PYTHON=python3
    fi
fi

exec "$PYTHON" -m pydlshogi2.player.native_mcts_player "$@"
