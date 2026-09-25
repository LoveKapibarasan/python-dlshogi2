"""Generate training data by letting a strong USI engine play itself.

The network in this repository learns from ``HuffmanCodedPosAndEval`` (hcpe)
records: a position, the move to imitate, a search evaluation and the game's
result.  Floodgate game records supply that from human-run engines of mixed
strength; this script supplies it from one engine we choose — typically
YaneuraOu with a Suisho evaluation — at a fixed search budget, which is a far
stronger and more consistent teacher than the average Floodgate game.

Each game starts with a few uniformly random moves so that games differ, then
the engine plays both sides with ``go nodes N``.  Every searched position is
recorded with the engine's best move and its score (stored from black's point of view,
like every other hcpe producer here);
once the game ends, every record gets the result.  Games are adjudicated when
the score passes ``--adjudicate`` so the engine does not spend time proving a
won position.

Run one process per core, each with its own output file and seed:

.. code-block:: bash

    for i in 0 1 2 3; do
        python utils/gen_teacher.py ~/src/YaneuraOu/yaneuraou.sh teacher-$i.hcpe \\
            --options EvalDir=/home/user/src/YaneuraOu/eval --nodes 30000 \\
            --games 100000 --seed $i &
    done
"""
import argparse
import os
import random
import subprocess
import sys
import time

import numpy as np
from cshogi import (Board, HuffmanCodedPosAndEval, BLACK, BLACK_WIN, WHITE_WIN,
                    DRAW, NOT_REPETITION, REPETITION_DRAW, REPETITION_WIN,
                    REPETITION_SUPERIOR, move16, move_to_usi)

MATE_SCORE = 30000


class UsiEngine:
    """A minimal USI client: enough to set options and run ``go nodes``."""

    def __init__(self, cmd, options):
        self.proc = subprocess.Popen([cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.send('usi')
        self.wait_for('usiok')
        for name, value in options.items():
            self.send('setoption name {} value {}'.format(name, value))
        self.send('isready')
        self.wait_for('readyok')

    def send(self, line):
        self.proc.stdin.write(line + '\n')
        self.proc.stdin.flush()

    def wait_for(self, token):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise EOFError('engine exited')
            if line.strip() == token:
                return

    def go(self, moves, nodes):
        """Search the position after ``moves`` and return ``(bestmove, score)``.

        ``score`` is from the side to move, in centipawns, with mates mapped to
        ``+-MATE_SCORE``.  ``None`` if the engine printed no score.
        """
        self.send('position startpos moves ' + ' '.join(moves) if moves else 'position startpos')
        self.send('go nodes {}'.format(nodes))
        score = None
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise EOFError('engine exited')
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == 'info' and 'score' in tokens:
                i = tokens.index('score')
                kind, value = tokens[i + 1], tokens[i + 2]
                if kind == 'cp':
                    score = int(value)
                elif kind == 'mate':
                    # "mate +" / "mate -" (手数不明) にも対応する
                    score = -MATE_SCORE if value.startswith('-') else MATE_SCORE
            elif tokens[0] == 'bestmove':
                return tokens[1], score

    def quit(self):
        try:
            self.send('quit')
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def play_game(engine, rng, args):
    """Play one game; return a list of ``(hcp, eval, move16, turn)`` and the result."""
    board = Board()
    moves = []
    records = []

    # 序盤のランダム手で対局を散らす
    random_plies = rng.randint(args.random_min, args.random_max)
    for _ in range(random_plies):
        legal = list(board.legal_moves)
        if not legal:
            break
        move = rng.choice(legal)
        board.push(move)
        moves.append(move_to_usi(move))
    if board.is_game_over():
        return [], None

    result = DRAW
    while board.move_number <= args.max_ply:
        if board.is_game_over():
            result = WHITE_WIN if board.turn == BLACK else BLACK_WIN
            break
        draw = board.is_draw()
        if draw == REPETITION_DRAW:
            result = DRAW
            break
        if draw != NOT_REPETITION:
            # 連続王手の千日手・優等/劣等局面: 手番側から見た勝敗
            win = draw in (REPETITION_WIN, REPETITION_SUPERIOR)
            result = BLACK_WIN if (board.turn == BLACK) == win else WHITE_WIN
            break

        bestmove, score = engine.go(moves, args.nodes)
        if bestmove == 'resign':
            result = WHITE_WIN if board.turn == BLACK else BLACK_WIN
            break
        if bestmove == 'win':
            result = BLACK_WIN if board.turn == BLACK else WHITE_WIN
            break
        move = board.move_from_usi(bestmove)

        if score is not None:
            hcp = np.empty(1, dtype=HuffmanCodedPosAndEval)
            board.to_hcp(hcp['hcp'])
            # hcpe の eval は先手視点 (csa_to_hcpe・selfplay と同じ規約)
            score_black = score if board.turn == BLACK else -score
            records.append((hcp['hcp'][0].copy(), max(-32000, min(32000, score_black)),
                            move16(move), board.turn))
            if abs(score) >= args.adjudicate:
                side_wins = score > 0
                result = BLACK_WIN if (board.turn == BLACK) == side_wins else WHITE_WIN
                break

        board.push(move)
        moves.append(bestmove)

    return records, result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('engine', help='USI engine launcher')
    parser.add_argument('output', help='hcpe file to append to')
    parser.add_argument('--options', default='',
                        help='USI options as name=value,name=value')
    parser.add_argument('--nodes', type=int, default=30000, help='search budget per move')
    parser.add_argument('--games', type=int, default=1000)
    parser.add_argument('--random-min', type=int, default=0)
    parser.add_argument('--random-max', type=int, default=8)
    parser.add_argument('--max-ply', type=int, default=320,
                        help='declare a draw after this many plies')
    parser.add_argument('--adjudicate', type=int, default=3000,
                        help='end the game once |score| reaches this (cp)')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    options = {'Threads': 1, 'USI_Hash': 64, 'USI_OwnBook': 'false', 'BookFile': 'no_book',
               'NetworkDelay': 0, 'NetworkDelay2': 0, 'ResignValue': 99999}
    for item in filter(None, args.options.split(',')):
        name, value = item.split('=', 1)
        options[name] = value

    rng = random.Random(args.seed)
    engine = UsiEngine(os.path.abspath(args.engine), options)
    start = time.time()
    total = 0
    try:
        for game in range(1, args.games + 1):
            engine.send('usinewgame')
            records, result = play_game(engine, rng, args)
            if not records:
                continue
            data = np.zeros(len(records), dtype=HuffmanCodedPosAndEval)
            for i, (hcp, score, bestmove16, _turn) in enumerate(records):
                data[i]['hcp'] = hcp
                data[i]['eval'] = score
                data[i]['bestMove16'] = bestmove16
                data[i]['gameResult'] = result
            with open(args.output, 'ab') as f:
                data.tofile(f)
            total += len(records)
            if game % 10 == 0:
                elapsed = time.time() - start
                print('game {} positions {} ({:.1f}/s)'.format(game, total, total / elapsed),
                      flush=True)
    finally:
        engine.quit()


if __name__ == '__main__':
    main()
