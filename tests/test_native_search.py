"""Proves the native (C++) search is the Python search, operation for operation.

``pydlshogi2/uct/native/uct.cpp`` ports ``MCTSPlayer.search`` and everything
under it.  The only thing that makes the port trustworthy is that, fed the same
network outputs, it builds the same tree: here both run a fixed number of
playouts from the same positions on the CPU (fp32, no CUDA graph) and must end
with identical root visit counts and the same best move.

Skipped when the library has not been built (``pydlshogi2/uct/native/build.sh``)
or the checkpoint is missing.
"""
import contextlib
import io
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CHECKPOINT = os.path.join(ROOT, 'checkpoints', 'checkpoint.pth')
LIBRARY = os.path.join(ROOT, 'pydlshogi2', 'uct', 'native', 'libuct.so')

try:
    import numpy as np
    from pydlshogi2.player.mcts_player import MCTSPlayer
    from pydlshogi2.player.native_mcts_player import NativeMCTSPlayer
    DEPENDENCIES = None
except ImportError as error:  # torch / cshogi / numpy が無い環境
    DEPENDENCIES = str(error)

POSITIONS = [
    [],
    # 後手番で持ち駒あり (入力特徴量の持ち駒プレーンの並びを確かめる)
    '7g7f 3c3d 2g2f 8c8d 2f2e 8d8e 6i7h 4a3b 2e2d 2c2d 2h2d'.split(),
    '2g2f 8c8d 2f2e 8d8e 6i7h 4a3b 2e2d 2c2d 2h2d P*2c 2d2f'.split(),
]


def make(cls):
    player = cls()
    player.modelfile = CHECKPOINT
    player.gpu_id = -1
    player.pv_interval = 0
    if cls is NativeMCTSPlayer:
        player.fast_inference = False
    with contextlib.redirect_stdout(io.StringIO()):
        player.isready()
    return player


def search(player, moves, playouts):
    # 前の探索の木を使わないよう、別の開始局面を挟んで作り直させる
    player.position('sfen lnsgkgsnl/9/9/9/9/9/9/9/LNSGKGSNL b - 1', [])
    player.tree.history_starting_pos_key = None
    player.position('startpos', moves)
    player.set_limits(nodes=playouts)
    with contextlib.redirect_stdout(io.StringIO()):
        bestmove, _ = player.go()
    if isinstance(player, NativeMCTSPlayer):
        counts = player._refresh_root().child_move_count
    else:
        counts = player.tree.current_head.child_move_count
    return bestmove, np.asarray(counts).copy()


@unittest.skipIf(DEPENDENCIES, 'needs numpy/torch/cshogi: {}'.format(DEPENDENCIES))
@unittest.skipUnless(os.path.exists(CHECKPOINT), 'needs checkpoints/checkpoint.pth')
@unittest.skipUnless(os.path.exists(LIBRARY), 'native library not built (pydlshogi2/uct/native/build.sh)')
class NativeSearchEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        # CPU 推論が全コアを取って、他の作業 (対局など) を乱さないように
        torch.set_num_threads(2)
        cls.python = make(MCTSPlayer)
        cls.native = make(NativeMCTSPlayer)

    def test_same_tree_as_python(self):
        for moves in POSITIONS:
            with self.subTest(plies=len(moves)):
                best_py, counts_py = search(self.python, moves, 96)
                best_nat, counts_nat = search(self.native, moves, 96)
                self.assertEqual(best_py, best_nat)
                np.testing.assert_array_equal(counts_py, counts_nat)


if __name__ == '__main__':
    unittest.main()
