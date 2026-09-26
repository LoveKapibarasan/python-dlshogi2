#!/bin/sh
# Build the native UCT core (libuct.so) against cshogi's C++ sources.
#   CSHOGI_SRC  checkout of https://github.com/TadaoYamaoka/cshogi at the installed
#               version (default: ~/src/cshogi-src)
set -e
cd "$(dirname "$0")"
CSHOGI_SRC="${CSHOGI_SRC:-$HOME/src/cshogi-src}"
SRC="$CSHOGI_SRC/src"
# -ffp-contract=off: FMA で丸めが変わると、Python 版と同じ探索にならない
# -DNDEBUG: cshogi の assert (Position::isOK など) を外す。付けないと 1 手の
#           doMove/undoMove が 7us、3 手詰め探索が 21us かかる (付けると 0.12us / 2.7us)。
#           Python 拡張としてビルドされる cshogi 本体には setuptools が付けている
g++ -O3 -std=c++17 -fPIC -shared -march=native -ffp-contract=off -DNDEBUG \
    -DHAVE_SSE4 -DHAVE_SSE42 -DHAVE_AVX2 -I"$SRC" \
    uct.cpp "$SRC"/bitboard.cpp "$SRC"/common.cpp "$SRC"/generateMoves.cpp "$SRC"/hand.cpp \
    "$SRC"/init.cpp "$SRC"/move.cpp "$SRC"/mt64bit.cpp "$SRC"/position.cpp "$SRC"/search.cpp \
    "$SRC"/square.cpp "$SRC"/usi.cpp "$SRC"/book.cpp "$SRC"/dfpn.cpp "$SRC"/osl_dfpn.cpp \
    -pthread -o libuct.so
echo "built $(pwd)/libuct.so"
