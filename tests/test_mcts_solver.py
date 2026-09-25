"""Tests for the MCTS solver (EXP-002).

A node's ``value`` of ``VALUE_WIN`` / ``VALUE_LOSE`` means the side to move
there wins / loses by force.  :func:`propagate_proof` lifts such results from
children to their parent, and :meth:`MCTSPlayer.select_root_move` makes the
root obey them regardless of visit counts.

The node-level tests need no model.  The search-level test loads
``checkpoints/checkpoint.pth`` on the CPU and is skipped when it is missing.

Run with::

    python -m unittest discover -s tests
"""
import contextlib
import io
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    import numpy as np
    from cshogi import Board, move_to_usi
    from pydlshogi2.player.mcts_player import (
        MCTSPlayer, VALUE_WIN, VALUE_LOSE, VALUE_DRAW, propagate_proof,
    )
    from pydlshogi2.uct.uct_node import UctNode
    DEPENDENCIES = None
except ImportError as error:  # torch / cshogi / numpy が無い環境
    DEPENDENCIES = str(error)

CHECKPOINT = os.path.join(ROOT, 'checkpoints', 'checkpoint.pth')

# 後手番の3手詰め (1手詰めは無い)。ランダム対局から拾った局面で、
# 方策ネットワークは初手 9e8e を見つけられない
MATE_IN_3 = '7nb/+Bl4ss1/npp1pg2l/p2pGpp1p/r2G2kpP/1KPPl2R1/PPS1P1P1N/6G1L/1N1S5 w 2p 100'


class FakeBoard:
    """Just enough of ``cshogi.Board`` for :meth:`UctNode.expand_node`."""

    def __init__(self, moves):
        self.legal_moves = list(moves)


def make_parent(child_values, counts=None):
    """An expanded node whose children carry ``child_values`` (``None`` = not created)."""
    node = UctNode()
    node.expand_node(FakeBoard(range(len(child_values))))
    node.set_policy(np.full(len(child_values), 1 / len(child_values), dtype=np.float32))
    node.value = 0.5
    node.child_node = []
    for value in child_values:
        if value == 'absent':
            node.child_node.append(None)
        else:
            child = UctNode()
            child.value = value
            node.child_node.append(child)
    if counts is not None:
        node.child_move_count[:] = counts
        node.child_sum_value[:] = np.asarray(counts, dtype=np.float32) * 0.5
    return node


@unittest.skipIf(DEPENDENCIES, 'needs numpy/torch/cshogi: {}'.format(DEPENDENCIES))
class PropagateProofTest(unittest.TestCase):
    def test_one_losing_child_proves_a_win(self):
        node = make_parent([0.4, VALUE_LOSE, 'absent'])
        propagate_proof(node, 1)
        self.assertEqual(node.value, VALUE_WIN)

    def test_all_winning_children_prove_a_loss(self):
        node = make_parent([VALUE_WIN, VALUE_WIN, VALUE_WIN])
        propagate_proof(node, 2)
        self.assertEqual(node.value, VALUE_LOSE)

    def test_an_unexplored_child_blocks_the_loss(self):
        # 未作成の子は逃げ道かもしれない
        node = make_parent([VALUE_WIN, 'absent', VALUE_WIN])
        propagate_proof(node, 0)
        self.assertEqual(node.value, 0.5)

    def test_an_unproven_child_blocks_the_loss(self):
        node = make_parent([VALUE_WIN, 0.1, VALUE_WIN])
        propagate_proof(node, 0)
        self.assertEqual(node.value, 0.5)

    def test_draws_are_not_propagated(self):
        node = make_parent([VALUE_DRAW, VALUE_WIN])
        propagate_proof(node, 0)
        self.assertEqual(node.value, 0.5)
        node = make_parent([VALUE_DRAW, VALUE_DRAW])
        propagate_proof(node, 1)
        self.assertEqual(node.value, 0.5)

    def test_a_child_proven_lost_gets_an_exact_loss_value(self):
        node = make_parent([VALUE_WIN, 0.5], counts=[10, 4])
        propagate_proof(node, 0)
        self.assertEqual(node.child_sum_value[0], 0.0)
        self.assertEqual(node.child_q[0], 0.0)
        self.assertEqual(node.value, 0.5)

    def test_a_proven_node_stays_proven(self):
        node = make_parent([VALUE_WIN, VALUE_WIN])
        node.value = VALUE_WIN
        propagate_proof(node, 0)
        self.assertEqual(node.value, VALUE_WIN)


@unittest.skipIf(DEPENDENCIES, 'needs numpy/torch/cshogi: {}'.format(DEPENDENCIES))
class SelectRootMoveTest(unittest.TestCase):
    def setUp(self):
        self.player = MCTSPlayer.__new__(MCTSPlayer)

    def test_unproven_root_picks_most_visited(self):
        node = make_parent([0.5, 0.5, 0.5], counts=[3, 9, 4])
        self.assertEqual(self.player.select_root_move(node), (1, None))

    def test_forced_win_beats_visit_count(self):
        node = make_parent([0.5, 0.5, VALUE_LOSE], counts=[50, 40, 2])
        self.assertEqual(self.player.select_root_move(node), (2, 1.0))

    def test_proven_loss_is_never_played(self):
        node = make_parent([VALUE_WIN, 0.5, 0.5], counts=[90, 5, 7])
        self.assertEqual(self.player.select_root_move(node), (2, None))

    def test_only_unvisited_moves_left(self):
        # 訪問済みの手が全部負けと証明され、未訪問の手しか残っていない。
        # 以前は訪問数0の手の勝率を 0/0 で求めて例外になり、bestmove が返らなかった
        node = make_parent([VALUE_WIN, 'absent', 'absent'], counts=[90, 0, 0])
        node.set_policy(np.array([0.5, 0.1, 0.4], dtype=np.float32))
        self.assertEqual(self.player.select_root_move(node), (2, None))

    def test_all_lost_still_returns_a_move(self):
        node = make_parent([VALUE_WIN, VALUE_WIN], counts=[3, 8])
        self.assertEqual(self.player.select_root_move(node), (1, 0.0))


@unittest.skipIf(DEPENDENCIES, 'needs numpy/torch/cshogi: {}'.format(DEPENDENCIES))
@unittest.skipUnless(os.path.exists(CHECKPOINT), 'needs checkpoints/checkpoint.pth')
class SolverSearchTest(unittest.TestCase):
    def test_finds_mate_in_three_the_root_search_skips(self):
        player = MCTSPlayer()
        player.modelfile = CHECKPOINT
        player.gpu_id = -1
        # ルートの詰み探索を1手に絞り、3手詰めは木の中の solver だけが見つけられるようにする
        player.mate_root_ply = 1
        player.pv_interval = 0
        with contextlib.redirect_stdout(io.StringIO()):
            player.isready()
            player.position('sfen ' + MATE_IN_3, [])
            player.set_limits(nodes=300)
            bestmove, _ = player.go()

        self.assertEqual(player.tree.current_head.value, VALUE_WIN)
        # 選んだ手のあと、相手のどの応手にも詰みがあること (木の中の判定と同じ3手詰め)
        board = Board(MATE_IN_3)
        board.push_usi(bestmove)
        replies = list(board.legal_moves)
        self.assertTrue(replies)
        for reply in replies:
            board.push(reply)
            self.assertTrue(board.mate_move(3), move_to_usi(reply))
            board.pop()


if __name__ == '__main__':
    unittest.main()
