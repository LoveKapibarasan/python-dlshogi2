"""Proves the optimised PUCT selection picks the same child as the old one.

``MCTSPlayer.select_max_ucb_child`` was rewritten to read incrementally
maintained per-child values instead of recomputing them from the raw counts on
every descent.  That is a pure performance change, and the only thing that
makes it safe is that it selects *exactly* the same move — a bias here would
not crash anything, it would quietly make the engine weaker in a way no unit
test of the search would notice.

So this compares it against a transcription of the formula it replaced, over
randomly generated node states, including the awkward ones: no children
visited, all of them visited, virtual loss applied and rolled back.

Needs numpy, torch and cshogi (the player module imports them), so it skips on
a machine that only has the standard library.

Run with::

    python -m unittest discover -s tests
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    from pydlshogi2.player.mcts_player import MCTSPlayer, VIRTUAL_LOSS, update_result
    from pydlshogi2.uct.uct_node import UctNode
    DEPENDENCIES = None
except ImportError as error:  # torch / cshogi / numpy が無い環境
    DEPENDENCIES = str(error)


def reference_select(player, node):
    """The selection formula exactly as it was before the optimisation."""
    child_move_count = node.child_move_count

    if node.move_count == 0:
        return np.argmax(node.policy)

    q = np.divide(node.child_sum_value, child_move_count,
                  out=np.zeros(len(node.child_move), np.float32),
                  where=child_move_count != 0)
    visited = child_move_count != 0
    parent_q = node.sum_value / node.move_count
    visited_policy_sum = node.policy[visited].sum()
    fpu = parent_q - player.fpu_reduction * np.sqrt(visited_policy_sum, dtype=np.float32)
    q = np.where(visited, q, np.float32(fpu))

    c = math.log((node.move_count + player.c_base + 1) / player.c_base) + player.c_puct
    u = c * node.policy * np.sqrt(np.float32(node.move_count)) / (1 + child_move_count)
    ucb = q + u

    return np.argmax(ucb)


class FakeBoard:
    """Just enough of ``cshogi.Board`` for :meth:`UctNode.expand_node`."""

    def __init__(self, moves):
        self.legal_moves = list(moves)


def make_node(child_num, rng):
    """Build an expanded, evaluated node with a random prior."""
    node = UctNode()
    node.expand_node(FakeBoard(range(child_num)))
    policy = rng.random(child_num).astype(np.float32)
    node.set_policy(policy / policy.sum())
    return node


@unittest.skipIf(DEPENDENCIES, 'needs numpy/torch/cshogi: {}'.format(DEPENDENCIES))
class SelectionEquivalenceTest(unittest.TestCase):
    def setUp(self):
        # MCTSPlayer() を作るとモデルまで触りに行くので、探索定数だけ持つ器を使う
        self.player = MCTSPlayer.__new__(MCTSPlayer)
        self.player.c_puct = 1.0
        self.player.c_base = 19652.0
        self.player.fpu_reduction = 0.27

    def assert_same_choice(self, node):
        expected = int(reference_select(self.player, node))
        actual = int(self.player.select_max_ucb_child(node))
        if expected == actual:
            return
        # 浮動小数の丸めで首位が入れ替わったのか、本当に違う手を選んだのかを区別する
        reference = self.reference_scores(node)
        self.assertAlmostEqual(
            float(reference[expected]), float(reference[actual]), places=5,
            msg='selected child {} instead of {} at a non-tie'.format(actual, expected))

    def reference_scores(self, node):
        """The reference UCB vector, for diagnosing a disagreement."""
        player = self.player
        count = node.child_move_count
        q = np.divide(node.child_sum_value, count,
                      out=np.zeros(len(node.child_move), np.float32),
                      where=count != 0)
        visited = count != 0
        fpu = (node.sum_value / node.move_count
               - player.fpu_reduction * np.sqrt(node.policy[visited].sum(), dtype=np.float32))
        q = np.where(visited, q, np.float32(fpu))
        c = math.log((node.move_count + player.c_base + 1) / player.c_base) + player.c_puct
        return q + c * node.policy * np.sqrt(np.float32(node.move_count)) / (1 + count)

    def visit(self, node, index, result):
        """Apply virtual loss and back a result up, as the search does."""
        node.move_count += VIRTUAL_LOSS
        node.child_move_count[index] += VIRTUAL_LOSS
        node.refresh_child(index)
        update_result(node, index, result)

    def test_unvisited_root_follows_the_policy(self):
        rng = np.random.default_rng(0)
        node = make_node(30, rng)
        self.assertEqual(int(self.player.select_max_ucb_child(node)),
                         int(np.argmax(node.policy)))

    def test_agrees_over_a_random_search(self):
        rng = np.random.default_rng(1)
        for child_num in (2, 5, 40, 180):
            node = make_node(child_num, rng)
            for _ in range(400):
                self.assert_same_choice(node)
                index = int(rng.integers(child_num))
                self.visit(node, index, float(rng.random()))

    def test_agrees_when_every_child_is_visited(self):
        rng = np.random.default_rng(2)
        node = make_node(12, rng)
        for index in range(12):
            self.visit(node, index, float(rng.random()))
        for _ in range(200):
            self.assert_same_choice(node)
            self.visit(node, int(rng.integers(12)), float(rng.random()))

    def test_agrees_with_virtual_loss_in_flight(self):
        rng = np.random.default_rng(3)
        node = make_node(20, rng)
        for _ in range(30):
            self.visit(node, int(rng.integers(20)), float(rng.random()))

        # バッチの途中の状態: virtual loss だけが乗った子がいる
        pending = [int(rng.integers(20)) for _ in range(5)]
        for index in pending:
            node.move_count += VIRTUAL_LOSS
            node.child_move_count[index] += VIRTUAL_LOSS
            node.refresh_child(index)
        self.assert_same_choice(node)

        # 破棄された経路の巻き戻し
        for index in reversed(pending):
            node.move_count -= VIRTUAL_LOSS
            node.child_move_count[index] -= VIRTUAL_LOSS
            node.refresh_child(index)
        self.assert_same_choice(node)

    def test_cache_matches_a_full_recomputation(self):
        """The incremental values must equal what recomputing them would give."""
        rng = np.random.default_rng(4)
        node = make_node(25, rng)
        for step in range(300):
            self.visit(node, int(rng.integers(25)), float(rng.random()))
            if step % 25:
                continue
            count = node.child_move_count
            visited = count != 0
            expected_q = np.divide(node.child_sum_value, count,
                                   out=np.zeros(25, np.float32), where=visited)
            np.testing.assert_allclose(node.child_q, expected_q, rtol=1e-6)
            np.testing.assert_allclose(node.child_unvisited,
                                       (~visited).astype(np.float32))
            np.testing.assert_allclose(node.child_policy_denom,
                                       node.policy / (1 + count), rtol=1e-6)
            self.assertAlmostEqual(node.visited_policy_sum,
                                   float(node.policy[visited].sum()), places=5)

    def test_replacing_the_policy_rebuilds_the_cache(self):
        # 自己対局は探索前にルートの方策を Dirichlet ノイズで差し替える
        rng = np.random.default_rng(5)
        node = make_node(15, rng)
        for _ in range(20):
            self.visit(node, int(rng.integers(15)), float(rng.random()))

        noisy = rng.random(15).astype(np.float32)
        node.set_policy(noisy / noisy.sum())

        count = node.child_move_count
        self.assertAlmostEqual(node.visited_policy_sum,
                               float(node.policy[count != 0].sum()), places=5)
        np.testing.assert_allclose(node.child_policy_denom,
                                   node.policy / (1 + count), rtol=1e-6)
        self.assert_same_choice(node)

    def test_pruning_the_tree_clears_the_cache(self):
        rng = np.random.default_rng(6)
        node = make_node(8, rng)
        self.visit(node, 3, 0.5)
        node.child_node = [None] * 8
        node.release_children_except_one(3)
        self.assertIsNone(node.child_q)
        self.assertIsNone(node.child_unvisited)
        self.assertIsNone(node.child_policy_denom)
        self.assertEqual(node.visited_policy_sum, 0.0)


if __name__ == '__main__':
    unittest.main()
