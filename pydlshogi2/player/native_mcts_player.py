"""MCTS player whose search runs in the native UCT core.

:class:`NativeMCTSPlayer` is :class:`~pydlshogi2.player.mcts_player.MCTSPlayer`
with the tree search moved to C++ (``pydlshogi2/uct/native/uct.cpp``, loaded
through ctypes).  Everything around the search is inherited unchanged: the USI
protocol, the options, time management, the root mate checks and the
resignation rule.  The neural network also stays in Python — the native search
fills the feature buffer and calls :meth:`_evaluate` for each batch.

The native core ports the Python search operation for operation, so at a fixed
playout count the two choose the same moves (up to the last bit of the
softmax's ``exp``); what changes is how many playouts fit in a second.

Build the library first::

    pydlshogi2/uct/native/build.sh
"""
import ctypes
import math
import os
import time

import numpy as np
import torch
from cshogi import move_to_usi

from pydlshogi2.features import MOVE_LABELS_NUM
from pydlshogi2.player.mcts_player import MCTSPlayer, VALUE_WIN

LIBRARY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'uct', 'native', 'libuct.so')

EVAL_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)
INTERRUPT_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int)

# 合法手の最大数 (将棋は593手)
MAX_MOVES = 600


def load_library(path=LIBRARY):
    lib = ctypes.CDLL(path)
    p, i, d, f = ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_float
    fp = ctypes.POINTER(ctypes.c_float)
    ip = ctypes.POINTER(ctypes.c_int)
    lib.uct_new.argtypes = [i, fp, fp, fp]
    lib.uct_new.restype = p
    lib.uct_free.argtypes = [p]
    lib.uct_set_params.argtypes = [p, d, d, d, d, i]
    lib.uct_set_callbacks.argtypes = [p, EVAL_CALLBACK, INTERRUPT_CALLBACK]
    lib.uct_set_position.argtypes = [p, ctypes.c_char_p, ip, i]
    lib.uct_set_position.restype = i
    lib.uct_prepare_root.argtypes = [p]
    lib.uct_prepare_root.restype = i
    lib.uct_search.argtypes = [p, ctypes.c_longlong]
    lib.uct_search.restype = ctypes.c_longlong
    lib.uct_playout_count.argtypes = [p]
    lib.uct_playout_count.restype = ctypes.c_longlong
    lib.uct_root_stats.argtypes = [p, ip, ip, fp, i]
    lib.uct_root_stats.restype = i
    lib.uct_root_info.argtypes = [p, ip, ctypes.POINTER(d), fp]
    lib.uct_pv.argtypes = [p, i, ip, i]
    lib.uct_pv.restype = i
    return lib


def _ptr(array, ctype):
    return array.ctypes.data_as(ctypes.POINTER(ctype))


class _RootView:
    """The attributes of a root ``UctNode`` that ``MCTSPlayer`` reads."""

    def __init__(self, moves, counts, sums, move_count, sum_value, value):
        self.child_move = moves
        self.child_move_count = counts
        self.child_sum_value = sums
        self.move_count = move_count
        self.sum_value = sum_value
        self.value = value


class _TreeView:
    current_head = None


class NativeMCTSPlayer(MCTSPlayer):
    name = 'python-dlshogi2-native'

    def __init__(self):
        super().__init__()
        # fp16 + CUDA Graph で推論する (False なら MCTSPlayer と同じ fp32 の逐次実行)
        self.fast_inference = True
        self.lib = None
        self.handle = None
        self.native_tree = _TreeView()
        self._moves = []
        self._start = 'startpos'

    def usi(self):
        super().usi()
        print('option name fast_inference type check default true')

    def setoption(self, args):
        if args[1] == 'fast_inference':
            self.fast_inference = args[3] == 'true'
        else:
            super().setoption(args)

    # ---- 準備 ----
    def isready(self):
        super().isready()
        if self.lib is None:
            self.lib = load_library()
        if self.handle is not None:
            self.lib.uct_free(self.handle)
        self.graph = None
        if self.fast_inference and self.device.type == 'cuda':
            self._capture_graph()
        else:
            self.policy_out = np.zeros((self.batch_size, MOVE_LABELS_NUM), dtype=np.float32)
            self.value_out = np.zeros(self.batch_size, dtype=np.float32)
        self.features_np = self.features.numpy()
        self.handle = self.lib.uct_new(self.batch_size, _ptr(self.features_np, ctypes.c_float),
                                       _ptr(self.policy_out, ctypes.c_float),
                                       _ptr(self.value_out, ctypes.c_float))
        # コールバックは参照を保持しておかないと GC で消える
        self._eval_cb = EVAL_CALLBACK(self._evaluate)
        self._interrupt_cb = INTERRUPT_CALLBACK(self._interrupt)
        self.lib.uct_set_callbacks(self.handle, self._eval_cb, self._interrupt_cb)
        self._stats_moves = np.zeros(MAX_MOVES, dtype=np.int32)
        self._stats_counts = np.zeros(MAX_MOVES, dtype=np.int32)
        self._stats_sums = np.zeros(MAX_MOVES, dtype=np.float32)
        self._set_native_position()

    def _apply_params(self):
        self.lib.uct_set_params(self.handle, self.c_puct, self.c_base, self.fpu_reduction,
                                self.temperature, 3)

    def _capture_graph(self):
        """Capture one fp16, channels-last forward pass of a full batch as a CUDA graph.

        With a 10-block network and a batch of 32, eager PyTorch spends most
        of its time launching kernels one by one; replaying a captured graph
        launches them all at once, and fp16 in channels-last layout runs the
        convolutions on the tensor cores.  The outputs land in pinned host
        buffers that the native search reads directly.
        """
        model = self.model.half().to(memory_format=torch.channels_last)
        self.graph_input = torch.zeros((self.batch_size,) + tuple(self.features.shape[1:]),
                                       dtype=torch.float16, device=self.device
                                       ).to(memory_format=torch.channels_last)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                model(self.graph_input)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            policy, value = model(self.graph_input)
            self.graph_policy = policy.float()
            self.graph_value = torch.sigmoid(value.float()).reshape(-1)
        self.policy_pinned = torch.empty(self.graph_policy.shape, dtype=torch.float32, pin_memory=True)
        self.value_pinned = torch.empty(self.graph_value.shape, dtype=torch.float32, pin_memory=True)
        self.policy_out = self.policy_pinned.numpy()
        self.value_out = self.value_pinned.numpy()

    # ---- NN 評価 (C++ から呼ばれる) ----
    def _evaluate(self, n):
        if self.graph is not None:
            # バッチが満杯でなくても全体を流す (後ろの行は前回の残りで、結果は読まれない)
            self.graph_input.copy_(self.features, non_blocking=True)
            self.graph.replay()
            self.policy_pinned.copy_(self.graph_policy, non_blocking=True)
            self.value_pinned.copy_(self.graph_value, non_blocking=True)
            torch.cuda.current_stream().synchronize()
            return 0
        with torch.no_grad():
            x = self.features[0:n].to(self.device)
            policy_logits, value_logits = self.model(x)
            self.policy_out[:n] = policy_logits.cpu().numpy()
            self.value_out[:n] = torch.sigmoid(value_logits).cpu().numpy().reshape(-1)
        return 0

    def _interrupt(self):
        self.playout_count = self.lib.uct_playout_count(self.handle)
        if self.check_interruption():
            return 1
        if self.pv_interval > 0:
            elapsed_time = int((time.time() - self.begin_time) * 1000)
            if elapsed_time > self.last_pv_print_time + self.pv_interval:
                self.last_pv_print_time = elapsed_time
                self.get_bestmove_and_print_pv()
        return 0

    # ---- 局面 ----
    def position(self, sfen, usi_moves):
        # Python 側の root_board は、ルートでの詰み判定などに引き続き使う
        super().position(sfen, usi_moves)
        self._start = 'startpos' if sfen == 'startpos' else sfen[5:]
        self._moves = [m for m in self.root_board.history]
        if self.handle is not None:
            self._set_native_position()

    def _set_native_position(self):
        moves = np.asarray(self._moves, dtype=np.int32)
        self.lib.uct_set_position(self.handle, self._start.encode('ascii'),
                                  _ptr(moves, ctypes.c_int), len(moves))

    def _refresh_root(self):
        n = self.lib.uct_root_stats(self.handle, _ptr(self._stats_moves, ctypes.c_int),
                                    _ptr(self._stats_counts, ctypes.c_int),
                                    _ptr(self._stats_sums, ctypes.c_float), MAX_MOVES)
        move_count = ctypes.c_int()
        sum_value = ctypes.c_double()
        value = ctypes.c_float()
        self.lib.uct_root_info(self.handle, ctypes.byref(move_count), ctypes.byref(sum_value),
                               ctypes.byref(value))
        root = _RootView(self._stats_moves[:n].tolist(), self._stats_counts[:n].copy(),
                         self._stats_sums[:n].copy(), move_count.value, sum_value.value,
                         value.value)
        self.native_tree.current_head = root
        return root

    def check_interruption(self):
        # MCTSPlayer.check_interruption は self.tree.current_head を読むので、
        # その間だけネイティブのルートを見せる
        self._refresh_root()
        python_tree = self.tree
        self.tree = self.native_tree
        try:
            return super().check_interruption()
        finally:
            self.tree = python_tree

    # ---- 探索 ----
    def go(self):
        self.begin_time = time.time()

        if self.root_board.is_game_over():
            return 'resign', None
        if self.root_board.is_nyugyoku():
            return 'win', None

        root = self._refresh_root()
        if root.value == VALUE_WIN:
            matemove = self.root_board.mate_move(3)
            if matemove != 0:
                print('info score mate 3 pv {}'.format(move_to_usi(matemove)), flush=True)
                return move_to_usi(matemove), None
        if not self.root_board.is_check():
            matemove = self.root_board.mate_move_in_1ply()
            if matemove:
                print('info score mate 1 pv {}'.format(move_to_usi(matemove)), flush=True)
                return move_to_usi(matemove), None
            if self.mate_root_ply >= 3:
                matemove = self.root_board.mate_move(self.mate_root_ply)
                if matemove:
                    print('info score mate {} pv {}'.format(self.mate_root_ply, move_to_usi(matemove)), flush=True)
                    return move_to_usi(matemove), None

        self.playout_count = 0
        self._apply_params()
        n = self.lib.uct_prepare_root(self.handle)

        if self.halt is None and n == 1:
            root = self._refresh_root()
            if root.child_move_count[0] > 0:
                bestmove, bestvalue, ponder_move = self.get_bestmove_and_print_pv()
                return move_to_usi(bestmove), move_to_usi(ponder_move) if ponder_move else None
            return move_to_usi(root.child_move[0]), None

        self.last_pv_print_time = 0
        self.lib.uct_search(self.handle, -1)
        self.playout_count = self.lib.uct_playout_count(self.handle)

        bestmove, bestvalue, ponder_move = self.get_bestmove_and_print_pv()
        if bestvalue < self.resign_threshold:
            return 'resign', None
        return move_to_usi(bestmove), move_to_usi(ponder_move) if ponder_move else None

    def get_bestmove_and_print_pv(self):
        finish_time = time.time() - self.begin_time
        root = self._refresh_root()
        selected_index = int(np.argmax(root.child_move_count))
        bestvalue = root.child_sum_value[selected_index] / root.child_move_count[selected_index]
        bestmove = root.child_move[selected_index]

        if bestvalue == 1.0:
            cp = 30000
        elif bestvalue == 0.0:
            cp = -30000
        else:
            cp = int(-math.log(1.0 / bestvalue - 1.0) * 600)

        pv_moves = np.zeros(64, dtype=np.int32)
        length = self.lib.uct_pv(self.handle, selected_index, _ptr(pv_moves, ctypes.c_int), 64)
        pv = ' '.join(move_to_usi(int(m)) for m in pv_moves[:length])
        ponder_move = int(pv_moves[1]) if length > 1 else None

        print('info nps {} time {} nodes {} score cp {} pv {}'.format(
            int(self.playout_count / finish_time) if finish_time > 0 else 0,
            int(finish_time * 1000),
            root.move_count,
            cp, pv), flush=True)

        return bestmove, bestvalue, ponder_move


if __name__ == '__main__':
    player = NativeMCTSPlayer()
    player.run()
