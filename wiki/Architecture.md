# Architecture

The network is a policy-value ResNet: one shared convolutional trunk feeding two
heads — a distribution over moves, and a scalar win probability. Source:
[`pydlshogi2/features.py`](https://github.com/LoveKapibarasan/python-dlshogi2/blob/main/pydlshogi2/features.py)
and
[`pydlshogi2/network/policy_value_resnet.py`](https://github.com/LoveKapibarasan/python-dlshogi2/blob/main/pydlshogi2/network/policy_value_resnet.py).

## Input features — 104 planes on a 9×9 board

`FEATURES_NUM` is not a magic constant; it is computed from cshogi's own tables:

```python
FEATURES_NUM = len(PIECE_TYPES) * 2 + sum(MAX_PIECES_IN_HAND) * 2
```

| Group | Planes | What each plane holds |
|-------|-------:|-----------------------|
| Pieces on the board | 14 × 2 = 28 | One binary plane per piece type per side (14 types incl. promoted, own then opponent) |
| Pieces in hand | 38 × 2 = 76 | Counts unary-encoded: `MAX_PIECES_IN_HAND` = pawn 18, lance 4, knight 4, silver 4, gold 4, bishop 2, rook 2 → 38 per side |
| **Total** | **104** | |

Two details matter more than the count:

**Everything is from the side to move.** `make_input_features` calls
`board.piece_planes(features)` for black and `board.piece_planes_rotate(features)`
for white, and reverses the two hands. The network therefore never sees "black"
or "white" — only "me" and "the opponent", which halves what it has to learn and
lets a single value head mean "win probability for the side to move".

**Hand counts are unary, not one-hot.** For `num` pieces of a type, the first
`num` planes of that type's block are filled with 1:

```python
for num, max_num in zip(hands, MAX_PIECES_IN_HAND):
    features[i:i+num].fill(1)
    i += max_num
```

So "3 pawns in hand" lights up planes 1–3 of the 18-plane pawn block. This makes
"at least N in hand" a single feature the convolution can read directly, rather
than something it must reconstruct from a one-hot index.

## Move labels — 2,187 outputs

```python
MOVE_PLANES_NUM = len(MOVE_DIRECTION) + len(HAND_PIECES)   # 20 + 7 = 27
MOVE_LABELS_NUM = MOVE_PLANES_NUM * 81                     # = 2187
```

A move is encoded as **(plane, destination square)** — the destination, not the
origin:

- **20 direction planes** — 10 directions × {no promotion, promotion}. The
  directions are the 8 compass moves plus the two knight jumps
  (`UP2_LEFT`, `UP2_RIGHT`). Promotion is encoded by adding 10 to the direction
  index, which is why `MOVE_DIRECTION` lists the plain directions first and the
  `*_PROMOTE` ones in the same order.
- **7 drop planes** — one per hand piece type.

The label is `move_direction * 81 + to_sq`. Because a shogi piece can only reach
a square from one direction (given the direction *and* the destination, the
origin is determined for every piece except when two pieces of different types
could make the same directional move — which the destination-plus-direction pair
still disambiguates for legal-move filtering), this encoding is complete without
needing 81 × 81 outputs.

As with the input, the board is rotated for white (`to_sq = 80 - to_sq`), so the
policy head is also side-to-move relative.

At inference time the 2,187 logits are **filtered down to the legal moves** and
only then softmaxed — see [MCTS](MCTS#policy-evaluation). Illegal moves never
compete for probability mass.

## The trunk and the heads

```
input (B, 104, 9, 9)
  └─ conv 3×3 → BatchNorm → ReLU                       (channels, default 256)
  └─ N × ResNetBlock                                    (default 20)
       conv 3×3 → BN → ReLU → conv 3×3 → BN → [SE] → +skip → ReLU
  ├─ policy head:  conv 1×1 → 27 planes → flatten(2187) → learnable bias
  └─ value head:   conv 1×1 → 27 planes → BN → ReLU → flatten → FC(fcl) → ReLU → FC(1)
```

Notes on the pieces that are easy to misread:

- **No padding loss.** Every conv is `3×3, padding=1` or `1×1`, so the 9×9
  geometry survives the whole trunk. There is no pooling anywhere; the value
  head reaches a scalar through the fully-connected layers instead.
- **`Bias` on the policy head.** The policy conv has `bias=False`; a separate
  learnable bias of shape `(2187,)` is added *after* flattening. That is a
  per-(direction, square) bias rather than a per-plane one — 2,187 parameters
  instead of 27.
- **The value head outputs a logit.** `forward` returns the raw scalar; the
  trainer feeds it to `BCEWithLogitsLoss` and the players apply `sigmoid`. Do
  not sigmoid it twice.
- **Squeeze-and-Excitation** (`--no_se` to disable) global-average-pools each
  channel, passes it through a bottleneck MLP (`channels // 8`) and uses the
  sigmoid output to rescale the channels. It is cheap and reliably worth a small
  amount of strength in AlphaZero-style networks.

## Configurable architecture, embedded in the checkpoint

`--blocks` / `--channels` / `--fcl` / `--no_se` set the architecture, and the
resulting config dict is saved *inside* every checkpoint:

```python
checkpoint = {'epoch': ..., 't': ..., 'model': ..., 'optimizer': ..., 'network': network_config}
```

`load_network` reads it back, so players and the ONNX exporter rebuild the exact
network without being told the shape. Checkpoints saved before this existed have
no `'network'` key and fall back to `LEGACY_NETWORK_CONFIG` — `10 × 192`, no SE.

This is also why `--blocks` and friends are **ignored when `--resume` is given**:
the checkpoint's own architecture wins, because loading its weights into a
different shape could not work anyway.

---

See also: [MCTS](MCTS), [Training Pipeline](Training-Pipeline)
