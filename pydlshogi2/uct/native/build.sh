#!/bin/sh
# Build the native UCT core (libuct.so) against cshogi's C++ sources.
#   CSHOGI_SRC  checkout of https://github.com/TadaoYamaoka/cshogi at the installed
#               version (default: ~/src/cshogi-src)
set -e
cd "$(dirname "$0")"
CSHOGI_SRC="${CSHOGI_SRC:-$HOME/src/cshogi-src}"
SRC="$CSHOGI_SRC/src"
# -ffp-contract=off: FMA で丸めが変わると、Python 版と同じ探索にならない
g++ -O3 -std=c++17 -fPIC -shared -march=native -ffp-contract=off \
    -DHAVE_SSE4 -DHAVE_SSE42 -DHAVE_AVX2 -I"$SRC" \
    uct.cpp "$SRC"/bitboard.cpp "$SRC"/common.cpp "$SRC"/generateMoves.cpp "$SRC"/hand.cpp \
    "$SRC"/init.cpp "$SRC"/move.cpp "$SRC"/mt64bit.cpp "$SRC"/position.cpp "$SRC"/search.cpp \
    "$SRC"/square.cpp "$SRC"/usi.cpp "$SRC"/book.cpp "$SRC"/dfpn.cpp "$SRC"/osl_dfpn.cpp \
    -o libuct.so
echo "built $(pwd)/libuct.so"
