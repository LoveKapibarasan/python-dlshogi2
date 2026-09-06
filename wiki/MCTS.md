# MCTS

The search is AlphaZero-style PUCT over a tree of `UctNode`s, with batched
network evaluation. Source:
[`pydlshogi2/player/mcts_player.py`](https://github.com/LoveKapibarasan/python-dlshogi2/blob/main/pydlshogi2/player/mcts_player.py)
and
[`pydlshogi2/uct/uct_node.py`](https://github.com/LoveKapibarasan/python-dlshogi2/blob/main/pydlshogi2/uct/uct_node.py).

## One playout

1. Descend from the root, at each node picking the child that maximises the PUCT
   score (`select_max_ucb_child`), applying **virtual loss** on the way down.
2. On reaching an unexpanded node, expand it (list the legal moves) and **queue**
   it for evaluation — the playout returns the `QUEUING` sentinel instead of a
   value.
3. When the queue reaches `batchsize`, run one batched forward pass
   (`eval_node`) and back up every queued result.
4. Terminal positions short-circuit: `VALUE_WIN` / `VALUE_LOSE` / `VALUE_DRAW`
   are assigned without touching the network.

Batching is what makes the GPU worth using at all: one 9×9×104 position per
forward pass would leave the device almost idle. The cost is that the tree must
tolerate several playouts in flight at once, which is what virtual loss is for.

## Virtual loss

Before descending into a child, the searcher does:

```python
current_node.move_count += VIRTUAL_LOSS
current_node.child_move_count[next_index] += VIRTUAL_LOSS
```

The extra visits with no accompanying value temporarily *lower* that child's
mean value, so the next playout in the same batch is pushed towards a different
move instead of piling onto the same leaf. `update_result` undoes it when the
real value arrives:

```python
current_node.move_count += 1 - VIRTUAL_LOSS
current_node.sum_value += result
```

When a queued node is discarded (`DISCARDED`), the virtual loss is subtracted
back out explicitly. `VIRTUAL_LOSS = 1` here — larger values spread the batch
more aggressively at the cost of search quality.

## PUCT and FPU

```python
c = math.log((node.move_count + c_base + 1) / c_base) + c_puct
u = c * node.policy * sqrt(node.move_count) / (1 + child_move_count)
ucb = q + u
```

Two deviations from the textbook formula:

**The exploration constant grows with visits.** `c_base = 19652` (AlphaZero's
value) makes `c` creep up logarithmically as a node accumulates visits, so a node
that has been searched a lot keeps some willingness to try alternatives instead
of locking onto its first preference.

**FPU (First Play Urgency).** An unvisited child has no `q` to average. Using
`0` would make every unvisited move look like a certain loss; using the parent's
value would make them all look equally good. Instead:

```python
fpu = parent_q - fpu_reduction * sqrt(visited_policy_sum)
```

Unvisited children are seeded with the parent's mean value, reduced in
proportion to how much policy mass has *already been explored*. Early in a
node's life the reduction is small (try things); once the high-prior moves have
been visited, the penalty grows and the search stops wandering into low-prior
moves. `fpu_reduction` defaults to `0.27` and is exposed as a USI option
(as an integer percentage: `27`).

The very first visit to a node skips all of this and follows the raw policy
(`if node.move_count == 0: return node.policy.argmax()`), since every `q` is
still zero.

## なぜ選択がキャッシュを読むだけになっているのか

`select_max_ucb_child` の中身は、式のとおりには書かれていない。理由は測ってみると分かる。

同じ木の状態に対して `timeit` で直接計った、最適化前の 1 回あたりの時間:

| 合法手の数 | 最適化前 | 最適化後 | `refresh_child` |
|-----------|---------|---------|----------------|
| 12 | 141.7 us | 17.0 us | 1.8 us |
| 40 | 155.1 us | 19.1 us | 1.8 us |
| 120 | 144.0 us | 14.5 us | 1.4 us |

**合法手の数がほとんど効いていない**ことに注目してほしい。120 手あっても 12 手でも
同じ時間がかかる。これは計算量がボトルネックではないという意味で、実際に効いていたのは
**numpy の呼び出しオーバーヘッド**だった。40 要素程度の配列に対する numpy の 1 演算は、
実際の計算よりディスパッチのほうがずっと高くつく。最適化前の実装は 1 回の選択につき
12 回それを呼び、そのうち `node.policy[visited].sum()` (fancy indexing) と `np.where` が
特に重かった。

探索全体に占める割合も測ってある (`eval_node` の前後で `time.process_time()` を読む):

| | 最適化前 | 最適化後 |
|---|---------|---------|
| 木の操作 (選択・展開・バックアップ) | 79 % | 57〜62 % |
| ニューラルネット評価 (`eval_node`) | 21 % | 38〜43 % |

そこで、式は変えずに**呼ぶ回数**を減らした。UCB に必要な子ごとの量は、
どれも 1 手ぶんの訪問で 1 要素しか変わらない。したがって毎回作り直す必要はない:

| キャッシュ | 中身 | 更新するタイミング |
|-----------|------|-------------------|
| `child_q` | 子の平均価値 (未訪問は 0) | 訪問回数か価値が変わったとき |
| `child_unvisited` | 未訪問なら 1.0 | 初訪問と、その巻き戻し |
| `child_policy_denom` | `policy / (1 + 訪問回数)` | 訪問回数が変わったとき |
| `visited_policy_sum` | 訪問済みの子の方策の合計 | 初訪問と、その巻き戻し |

残った選択の処理は 4 回の配列演算と `argmax` だけになる:

```python
ucb = child_unvisited * fpu        # 未訪問の子に FPU を与える
ucb += child_q                     # 訪問済みの子は平均価値
ucb += scale * child_policy_denom  # 探索項
return ucb.argmax()
```

### 差分更新は 3 箇所すべてから呼ばなければならない

`refresh_child` を呼び忘れても**落ちない**。キャッシュが古いまま選択に使われ、
探索が静かに歪むだけで、エラーは出ない。呼び出し箇所は次の 3 つだけに限定してある:

1. `uct_search` — 降りるときの virtual loss 加算
2. `search` — 破棄した経路の virtual loss 巻き戻し
3. `update_result` — 結果のバックアップ

自己対局が Dirichlet ノイズで方策そのものを差し替える経路もあるので、そこでは
`UctNode.set_policy` がキャッシュを丸ごと作り直す。

`tests/test_select_max_ucb_child.py` が、旧実装をそのまま書き写したものと
新実装の選択が一致することを、ランダムな木の状態 (未訪問だけ・全訪問・
virtual loss が飛んでいる途中・巻き戻し後) で確認している。

### なぜ Python のスカラを使っているのか

`refresh_child` が `arr[i]` ではなく `arr.item(i)` を使っているのは意図的である。
`arr[i]` は numpy スカラ**オブジェクト**を作り、その後の四則演算 1 回ごとに
1 マイクロ秒近いディスパッチが乗る。最初の実装はそれで 1 回 16 マイクロ秒かかっていた。
`item()` で素の Python の数値にし、方策を Python の list にも写しておくことで、
同じ処理が 2 マイクロ秒になった。

### 計測についての注意

この節の数字を出すのに、2 つの落とし穴を踏んだ。記録しておく。

**cProfile の絶対値を信用しない。** cProfile は 1 回の呼び出しごとに自身の
オーバーヘッドを**呼ばれた側の tottime に計上する**。`select_max_ucb_child` は
1 回の探索で 34,000 回以上呼ばれるので、プロファイルの秒数はその分だけ膨らむ。
方向 (どの関数が重いか) は正しかったが、倍率を語るには `timeit` で直接計るべきだった。

**共有マシンでは壁時計を信用しない。** 作業台の GPU/CPU は ollama と共有で、
負荷は数分単位で変わる。同じ実装の同じ探索が 6.5 秒から 11.9 秒まで振れる。
最初に取った A/B は「最適化後のほうが 23 % 遅い」という結論を出したが、
これは単に測定中に負荷が乗っただけだった。`time.process_time()` に切り替え、
さらに BASE と FAST を交互に走らせて**最小値**で比べて初めて、
一貫した比 (1.5〜2.4 倍) が出た。

## Policy evaluation

`eval_node` does **not** softmax the 2,187 logits. It gathers only the logits of
the node's legal moves and softmaxes those:

```python
legal_move_probabilities[j] = policy_logit[make_move_label(move, color)]
probabilities = softmax_temperature_with_normalize(legal_move_probabilities, temperature)
```

The `temperature` USI option flattens (`> 1`) or sharpens (`< 1`) that
distribution. It affects the prior only — the visit counts still decide the move.

## Mate detection

Three layers, cheapest first, all in `go()` before the search starts:

| Check | When |
|-------|------|
| `mate_move_in_1ply()` | Always (unless in check) — a 1-ply mate is nearly free to find |
| `mate_move(mate_root_ply)` | Once at the root, if `mate_root_ply >= 3` (default **7**) |
| `mate_move(3)` | Inside the search, on nodes reached during descent |

木の内側が 3 手詰めに限られているのは速度のためで、深くする案は
[Improvement Backlog](Improvement-Backlog) の EXP-008 にある。

A found mate returns immediately with `info score mate N` — no playouts spent.
Raising `mate_root_ply` finds deeper forced wins but costs a fixed search at
every move; the option accepts up to 31.

Draws and nyugyoku are handled in the same place: `is_nyugyoku()` returns `win`,
and repetition results map to `VALUE_DRAW`, `VALUE_WIN` or `VALUE_LOSE`
depending on whether the repetition is a plain `sennichite` or a perpetual-check
win/loss.

## Tree reuse

`NodeTree.reset_to_position` is called on every `position` command. If the new
position is reachable from the old one by the moves given, it walks down the
tree, calling `release_children_except_one` at each step, and keeps the subtree
that was already searched — so the opponent's thinking time is not wasted.

The `seen_old_head` bookkeeping guards the case where the new head is an
*ancestor* of the previously searched node (e.g. a takeback): the retained
statistics would then be from a different position, so the node is reset.

## Ponder

Pondering runs the same search with `halt = 2**31-1`, i.e. it never stops on its
own. `stop` ends it; `ponderhit` converts it into a normal search by re-applying
the real time limits from the stored `last_limits`.

## Time management

`set_limits` computes:

```python
time_limit = remaining_time / (14 + max(0, 30 - move_number)) + inc
```

so the opening spends roughly 1/44 of the clock per move, easing to 1/14 after
move 30. Byoyomi raises the floor: `minimum_time = byoyomi - byoyomi_margin`,
and the limit is never below it.

`check_interruption` stops early when the outcome is already decided — it
estimates how many playouts fit in the remaining time and, if the second-best
move cannot catch the best one in that many visits, returns immediately. It
extends instead (once per move, doubling `time_limit`) when after move 20 there
is clock to spare and the top two moves are still close — either within 1.5× on
visits, or the second has the better win rate.

| USI option | Default | Effect |
|------------|--------:|--------|
| `batchsize` | 32 | Nodes per network forward pass |
| `c_puct` | 100 (= 1.00) | PUCT exploration constant |
| `fpu_reduction` | 27 (= 0.27) | FPU penalty scale |
| `temperature` | 100 (= 1.00) | Policy softmax temperature |
| `c_base` | 19652 | Base of the logarithmic growth of `c` |
| `mate_root_ply` | 7 | Depth of the one-shot root mate search |
| `resign_threshold` | 1 (= 1%) | Win rate below which the engine resigns |
| `time_margin` | 1000 ms | Safety margin on the main clock |
| `byoyomi_margin` | 100 ms | Safety margin on byoyomi |
| `pv_interval` | 500 ms | How often `info pv` is printed (0 = never) |

Spin options that represent fractions are sent as integer percentages and
divided by 100 in `setoption`.

---

See also: [Architecture](Architecture), [USI Engine](USI-Engine),
[Reinforcement Learning](Reinforcement-Learning)

---

See also: [Evaluation and Rating](Evaluation-and-Rating),
[Improvement Backlog](Improvement-Backlog)
