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
(`if node.move_count == 0: return np.argmax(node.policy)`), since every `q` is
still zero.

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
