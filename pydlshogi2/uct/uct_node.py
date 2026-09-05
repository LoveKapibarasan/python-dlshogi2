import numpy as np

class UctNode:
    """One node of the search tree.

    Besides the counts a PUCT search obviously needs, the node caches the
    *derived* quantities the selection formula reads on every descent —
    each child's mean value, whether it is still unvisited, its prior divided
    by its visit count, and the total prior mass of the visited children.

    Recomputing those from the raw counts is what
    ``MCTSPlayer.select_max_ucb_child`` used to do, and profiling put it at
    **half of the entire search time**: not because the arithmetic is expensive
    but because a dozen numpy calls on 40-element arrays are almost all
    per-call overhead.  Each of these values changes for exactly one child at a
    time, so they are updated incrementally instead — see :meth:`refresh_child`.
    """

    def __init__(self):
        self.move_count = 0           # ノードの訪問回数
        self.sum_value = 0.0          # 勝率の合計
        self.child_move = None        # 子ノードの指し手(リスト)
        self.child_move_count = None  # 子ノードの訪問回数(ndarray)
        self.child_sum_value = None   # 子ノードの勝率の合計(ndarray)
        self.child_node = None        # 子ノード(リスト)
        self.policy = None            # 方策ネットワークの予測確率(ndarray)
        self.value = None             # 価値

        # 以下は上の値から導かれるキャッシュ。差分更新する (refresh_child)
        self.child_q = None            # 子の平均価値(未訪問は0)
        self.child_unvisited = None    # 未訪問なら1.0、訪問済みなら0.0
        self.child_policy_denom = None # policy / (1 + 訪問回数)
        self.visited_policy_sum = 0.0  # 訪問済みの子の方策の合計
        self.ucb = None                # UCB計算用のバッファ
        self.u = None                  # 探索項計算用のバッファ
        # 差分更新をPythonの数値だけで行うための写し (numpyスカラは1演算1マイクロ秒近くかかる)
        self.policy_list = None        # policy と同じ値の Python float のリスト
        self.visited_flags = None      # 訪問済みなら1 (bytearray)

    # 子ノード作成
    def create_child_node(self, index):
        self.child_node[index] = UctNode()
        return self.child_node[index]

    # ノードの展開
    def expand_node(self, board):
        """List the legal moves and allocate the per-child arrays."""
        self.child_move = list(board.legal_moves)
        child_num = len(self.child_move)
        self.child_move_count = np.zeros(child_num, dtype=np.int32)
        self.child_sum_value = np.zeros(child_num, dtype=np.float32)

        self.child_q = np.zeros(child_num, dtype=np.float32)
        self.child_unvisited = np.ones(child_num, dtype=np.float32)
        self.child_policy_denom = np.zeros(child_num, dtype=np.float32)
        self.visited_policy_sum = 0.0
        self.ucb = np.empty(child_num, dtype=np.float32)
        self.u = np.empty(child_num, dtype=np.float32)
        self.policy_list = None
        self.visited_flags = bytearray(child_num)

    def set_policy(self, policy):
        """Set the network's prior and rebuild everything derived from it.

        Used both when a node is first evaluated and when the prior is replaced
        wholesale, as self-play does when it mixes in Dirichlet noise.

        :param policy: prior probability per legal move (``float32`` ndarray).
        """
        self.policy = policy
        self.policy_list = policy.tolist()
        count = self.child_move_count
        denominator = np.empty(len(policy), dtype=np.float32)
        np.add(count, 1, out=denominator, casting='unsafe')
        self.child_policy_denom = policy / denominator
        visited = count != 0
        self.visited_flags = bytearray(visited.astype(np.uint8).tobytes())
        self.child_unvisited = (~visited).astype(np.float32)
        self.visited_policy_sum = float(policy[visited].sum())

    def refresh_child(self, index):
        """Bring one child's cached values back in line with its counts.

        Must be called after **every** change to ``child_move_count`` or
        ``child_sum_value``: applying virtual loss on the way down, rolling it
        back on a discarded playout, and backing a result up.  Getting this
        wrong does not crash — it quietly biases the search — so the callers are
        kept to those three places.

        Written entirely in Python scalars: ``ndarray.item()`` returns a plain
        ``float``/``int`` where ``ndarray[i]`` would build a numpy scalar, and
        arithmetic on those costs close to a microsecond each.  With this called
        several times per playout that difference is worth more than the code
        it costs.

        :param index: index of the child that changed.
        """
        count = self.child_move_count.item(index)
        prior = self.policy_list[index]
        if count:
            if not self.visited_flags[index]:
                self.visited_flags[index] = 1
                self.child_unvisited[index] = 0.0
                self.visited_policy_sum += prior
            self.child_q[index] = self.child_sum_value.item(index) / count
        else:
            if self.visited_flags[index]:
                self.visited_flags[index] = 0
                self.child_unvisited[index] = 1.0
                self.visited_policy_sum -= prior
            self.child_q[index] = 0.0
        self.child_policy_denom[index] = prior / (1 + count)

    # 1つを除くすべての子を削除する
    def release_children_except_one(self, move):
        if self.child_move and self.child_node:
            # 一つを残して削除する
            for i in range(len(self.child_move)):
                if self.child_move[i] == move:
                    if self.child_node[i] is None:
                        # 新しいノードを作成する
                        self.child_node[i] = UctNode()
                    # 子ノードを一つにする
                    if len(self.child_move) > 1:
                        self.child_move = [move]
                        self.child_move_count = None
                        self.child_sum_value = None
                        self.policy = None
                        self._clear_child_cache()
                        self.child_node = [self.child_node[i]]
                    return self.child_node[0]

        # 子ノードが見つからなかった場合、または子ノードが未展開、または子ノードリストが未初期化の場合
        self.child_move = [move]
        self.child_move_count = None
        self.child_sum_value = None
        self.policy = None
        self._clear_child_cache()
        # 子ノードのリストを初期化する
        self.child_node = [UctNode()]
        return self.child_node[0]

    def _clear_child_cache(self):
        """Drop the derived per-child arrays along with the counts they mirror."""
        self.child_q = None
        self.child_unvisited = None
        self.child_policy_denom = None
        self.visited_policy_sum = 0.0
        self.ucb = None
        self.u = None
        self.policy_list = None
        self.visited_flags = None

class NodeTree:
    def __init__(self):
        self.current_head = None
        self.gamebegin_node = None
        self.history_starting_pos_key = None

    # ゲーム木内の位置を設定し、サブツリーの再利用を試みる
    def reset_to_position(self, starting_pos_key, moves):
        if self.history_starting_pos_key != starting_pos_key:
            # 開始位置が異なる場合、ゲーム木を作り直す
            self.gamebegin_node = UctNode()
            self.current_head = self.gamebegin_node

        self.history_starting_pos_key = starting_pos_key

        # 開始位置から順に、子ノード一つだけ残して、それ以外を解放する
        old_head = self.current_head
        prev_head = None
        self.current_head = self.gamebegin_node
        seen_old_head = self.gamebegin_node == old_head
        for move in moves:
            prev_head = self.current_head
            # current_headに着手を追加する
            self.current_head = self.current_head.release_children_except_one(move)
            if old_head == self.current_head:
                seen_old_head = True

        # 古いヘッドが現れない場合は、以前に探索された位置の祖先である位置がある可能性があることを意味する
        # つまり、古い子が以前にトリミングされていても、current_headは古いデータを保持する可能性がある
        # その場合、current_headをリセットする必要がある
        if not seen_old_head and self.current_head != old_head:
            if prev_head:
                assert len(prev_head.child_move) == 1
                prev_head.child_node[0] = UctNode()
                self.current_head = prev_head.child_node[0]
            else:
                # 開始局面に戻った場合
                self.gamebegin_node = UctNode()
                self.current_head = self.gamebegin_node
