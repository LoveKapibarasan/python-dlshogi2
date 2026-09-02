"""Self-play game generation for reinforcement learning.

This module plays the engine against itself and writes the visited positions as
HCPE records, so the generated data plugs straight into :mod:`pydlshogi2.train`
without any loader changes.

For each played position it records:

* the position (``hcp``),
* the **greedy** move (``argmax`` of the MCTS visit counts) as ``bestMove16`` —
  a more stable policy target than the move actually played,
* the MCTS **root win rate** converted to centipawns as ``eval`` — used as a
  bootstrapped value target when training with ``val_lambda < 1``,
* the eventual ``gameResult`` once the game finishes.

The move actually played is sampled from the visit counts with a temperature
(and Dirichlet noise is mixed into the root prior) so games stay diverse, which
is what makes the data useful for policy improvement.

The search itself reuses :class:`~pydlshogi2.player.mcts_player.MCTSPlayer` —
batched inference, virtual loss, mate detection and tree reuse are inherited
unchanged.

Example
-------

.. code-block:: bash

    python -m pydlshogi2.selfplay checkpoints/checkpoint.pth selfplay.hcpe \\
        --games 1000 --playouts 800 --gpu 0
"""
import argparse
import math
import time

import numpy as np
from cshogi import (
    Board, HuffmanCodedPosAndEval, move16, move_to_usi,
    BLACK, BLACK_WIN, WHITE_WIN, DRAW,
    NOT_REPETITION, REPETITION_DRAW, REPETITION_WIN, REPETITION_LOSE,
)

from pydlshogi2.player.mcts_player import MCTSPlayer
from pydlshogi2.metrics import MetricsWriter, gpu_name

# 評価値クリッピング上限 (16bit格納のため)
EVAL_CLIP = 32767


def winrate_to_cp(winrate):
    """Convert a win rate in ``[0, 1]`` to a clipped centipawn score.

    Uses the same logistic mapping as the engine's PV output
    (``cp = -log(1/p - 1) * 600``).

    :param winrate: win probability for the side to move.
    :returns: integer centipawn score clipped to 16-bit range.
    """
    if winrate <= 0.0:
        return -EVAL_CLIP
    if winrate >= 1.0:
        return EVAL_CLIP
    cp = int(-math.log(1.0 / winrate - 1.0) * 600)
    return max(-EVAL_CLIP, min(EVAL_CLIP, cp))


class SelfPlayEngine(MCTSPlayer):
    """Drive :class:`MCTSPlayer` headlessly to generate self-play games.

    :param dirichlet_alpha: concentration of the Dirichlet noise added to the
        root prior.
    :param noise_eps: mixing weight of the Dirichlet noise (``0`` disables it).
    """

    def __init__(self, dirichlet_alpha=0.15, noise_eps=0.25):
        super().__init__()
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_eps = noise_eps
        # 自己対局では思考ログを抑制する
        self.pv_interval = 0

    def apply_root_dirichlet_noise(self, node):
        """Mix Dirichlet noise into a node's policy prior in place.

        :param node: the root :class:`~pydlshogi2.uct.uct_node.UctNode`.
        """
        if self.noise_eps <= 0.0:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(node.policy))
        node.policy = ((1.0 - self.noise_eps) * node.policy
                       + self.noise_eps * noise).astype(np.float32)

    def think(self, playouts):
        """Run a fixed-playout search at the current root and add root noise.

        Replicates the minimal setup of :meth:`MCTSPlayer.go` needed for search
        (expanding and evaluating the root) without its mate shortcuts or USI
        output, then runs :meth:`MCTSPlayer.search`.

        :param playouts: number of playouts to run.
        """
        current_node = self.tree.current_head

        # ルートの展開と評価
        if current_node.child_move is None:
            current_node.expand_node(self.root_board)
        if current_node.policy is None:
            self.current_batch_index = 0
            self.queue_node(self.root_board, current_node)
            self.eval_node()

        # ルートにDirichletノイズを加える
        self.apply_root_dirichlet_noise(current_node)

        # 探索
        self.playout_count = 0
        self.halt = playouts
        self.begin_time = time.time()
        self.last_pv_print_time = 0
        self.search()

    def root_winrate(self):
        """Return the root win rate for the side to move.

        Uses the visit-weighted mean value when available, else the raw network
        value.
        """
        node = self.tree.current_head
        if node.move_count > 0:
            return float(node.sum_value / node.move_count)
        return float(node.value)


def select_move(child_move, child_move_count, temperature):
    """Pick a move from MCTS visit counts.

    :param child_move: list of candidate moves.
    :param child_move_count: visit count per candidate (``ndarray``).
    :param temperature: ``0`` selects the most-visited move; higher values
        sample more uniformly.
    :returns: a ``(played_move, greedy_move)`` tuple, where ``greedy_move`` is
        always the most-visited move (used as the training target).
    """
    greedy_index = int(np.argmax(child_move_count))
    greedy_move = child_move[greedy_index]

    if temperature <= 0.0:
        return greedy_move, greedy_move

    counts = child_move_count.astype(np.float64)
    if counts.sum() == 0:
        return greedy_move, greedy_move
    powered = counts ** (1.0 / temperature)
    probabilities = powered / powered.sum()
    played_index = int(np.random.choice(len(child_move), p=probabilities))
    return child_move[played_index], greedy_move


def play_game(engine, playouts, max_moves, temperature, temp_cutoff):
    """Play a single self-play game.

    :param engine: a ready :class:`SelfPlayEngine`.
    :param playouts: playouts per move.
    :param max_moves: declare a draw after this many plies.
    :param temperature: sampling temperature before ``temp_cutoff``.
    :param temp_cutoff: ply after which moves are chosen greedily.
    :returns: a ``(records, game_result)`` tuple, where ``records`` is a list of
        ``(hcp_ndarray, bestmove16, eval_cp_black_pov)`` and ``game_result`` is a
        cshogi result constant.
    """
    board = Board()
    usi_moves = []
    records = []
    game_result = DRAW
    # to_hcp に渡す書き込み可能なhcpバッファ (フィールドビュー)
    hcp_buffer = np.zeros(1, HuffmanCodedPosAndEval)

    while True:
        # 終局判定 (手番側の視点)
        if board.is_game_over():
            # 手番側が詰み → 相手の勝ち
            game_result = WHITE_WIN if board.turn == BLACK else BLACK_WIN
            break
        if board.is_nyugyoku():
            game_result = BLACK_WIN if board.turn == BLACK else WHITE_WIN
            break
        draw = board.is_draw()
        if draw == REPETITION_DRAW:
            game_result = DRAW
            break
        if draw == REPETITION_WIN:
            game_result = BLACK_WIN if board.turn == BLACK else WHITE_WIN
            break
        if draw == REPETITION_LOSE:
            game_result = WHITE_WIN if board.turn == BLACK else BLACK_WIN
            break
        if board.move_number > max_moves:
            game_result = DRAW
            break

        # ルート局面を設定して探索
        engine.position('startpos', usi_moves)
        engine.think(playouts)

        node = engine.tree.current_head
        if node.child_move is None or len(node.child_move) == 0:
            game_result = WHITE_WIN if board.turn == BLACK else BLACK_WIN
            break

        # 着手選択 (温度はtemp_cutoffまで)
        temp = temperature if board.move_number <= temp_cutoff else 0.0
        played_move, greedy_move = select_move(node.child_move, node.child_move_count, temp)

        # 教師データを記録 (着手前の局面)
        board.to_hcp(hcp_buffer[0]['hcp'])
        winrate = engine.root_winrate()
        cp = winrate_to_cp(winrate)
        # evalはBLACK視点で格納する (csa_to_hcpeと同じ規約)
        eval_black = cp if board.turn == BLACK else -cp
        records.append((hcp_buffer[0]['hcp'].copy(), move16(greedy_move), eval_black))

        # 着手
        board.push(played_move)
        usi_moves.append(move_to_usi(played_move))

    return records, game_result


def main():
    """Parse arguments, generate self-play games and write an HCPE file."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('modelfile', help='checkpoint (.pth) to play with')
    parser.add_argument('output', help='output HCPE file')
    parser.add_argument('--games', type=int, default=100, help='number of games to play')
    parser.add_argument('--playouts', type=int, default=800, help='playouts per move')
    parser.add_argument('--max_moves', type=int, default=512, help='max plies before a draw')
    parser.add_argument('--temperature', type=float, default=1.0, help='sampling temperature')
    parser.add_argument('--temp_cutoff', type=int, default=30, help='ply after which moves are greedy')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.15, help='root Dirichlet alpha')
    parser.add_argument('--noise_eps', type=float, default=0.25, help='root Dirichlet mixing weight')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id (-1 for CPU)')
    parser.add_argument('--batchsize', type=int, default=32, help='inference batch size')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed (set a distinct value per parallel worker)')
    parser.add_argument('--metrics', default=None,
                        help='JSON Lines file to append per-game statistics and '
                             'run metadata to; read by dashboard/app.py')
    parser.add_argument('--iteration', type=int, default=None,
                        help='RL loop iteration this run belongs to (recorded in '
                             'the metrics file so the dashboard can group workers)')
    args = parser.parse_args()

    # 並列ワーカーが同一局を量産しないようシードを設定する
    if args.seed is not None:
        np.random.seed(args.seed)

    metrics = MetricsWriter(
        args.metrics, kind='selfplay', args=vars(args),
        extra={'iteration': args.iteration, 'worker': args.seed,
               'gpu_name': gpu_name(args.gpu)})

    engine = SelfPlayEngine(dirichlet_alpha=args.dirichlet_alpha, noise_eps=args.noise_eps)
    engine.modelfile = args.modelfile
    engine.gpu_id = args.gpu
    engine.batch_size = args.batchsize
    engine.isready()

    total_positions = 0
    result_counts = {BLACK_WIN: 0, WHITE_WIN: 0, DRAW: 0}
    played_games = 0
    started = time.time()
    with open(args.output, 'wb') as f:
        for g in range(args.games):
            game_started = time.time()
            records, game_result = play_game(
                engine, args.playouts, args.max_moves, args.temperature, args.temp_cutoff)
            game_elapsed = time.time() - game_started
            if not records:
                continue

            played_games += 1
            result_counts[game_result] = result_counts.get(game_result, 0) + 1
            metrics.metric(scope='game', game=g + 1, iteration=args.iteration,
                           worker=args.seed, moves=len(records),
                           game_result=int(game_result), seconds=game_elapsed,
                           positions=total_positions + len(records))

            hcpes = np.zeros(len(records), HuffmanCodedPosAndEval)
            for i, (hcp, bestmove16, eval_black) in enumerate(records):
                hcpes[i]['hcp'] = hcp
                hcpes[i]['bestMove16'] = bestmove16
                hcpes[i]['eval'] = eval_black
                hcpes[i]['gameResult'] = game_result
            hcpes.tofile(f)

            total_positions += len(records)
            print('game {}/{} moves={} result={} total_positions={}'.format(
                g + 1, args.games, len(records), game_result, total_positions), flush=True)

    elapsed = time.time() - started
    metrics.metric(scope='summary', iteration=args.iteration, worker=args.seed,
                   games=played_games, positions=total_positions,
                   black_wins=result_counts.get(BLACK_WIN, 0),
                   white_wins=result_counts.get(WHITE_WIN, 0),
                   draws=result_counts.get(DRAW, 0),
                   mean_moves=(total_positions / played_games) if played_games else 0.0,
                   seconds=elapsed,
                   games_per_hour=(played_games / elapsed * 3600.0) if elapsed > 0 else 0.0,
                   playouts=args.playouts)
    metrics.close(status='completed')

    print('done. games={} positions={}'.format(args.games, total_positions))


if __name__ == '__main__':
    main()
