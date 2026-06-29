# Human-like shogi AI (separate sub-project)

This directory is a **separate effort** from the strong self-play engine. The
goal here is *human-likeness*, not maximal strength: imitate how humans of a
given skill level play, in the spirit of [Maia Chess](https://maiachess.com/).

It does **not** touch the reinforcement-learning pipeline. It only reuses the
shared network (`pydlshogi2.network`) and feature encoding (`pydlshogi2.features`)
at training time; the data tooling here depends only on `cshogi`.

## 0. Shogi Wars KIF -> CSA (if your data is Shogi Wars .kif)

Shogi Wars exports are UTF-8 KIF with the player rank on `先手段級：` / `後手段級：`
lines and a bare `投了` ending that `cshogi` cannot interpret. `kif_to_csa.py`
handles all of this and unifies the corpus to CSA, mapping each dan/kyu rank to
an **ordinal** (higher = stronger) emitted as a floodgate-style rate line:

```
30級=1, 29級=2, ..., 1級=30, 初段=31, 二段=32, ..., 九段=39
```

```bash
python human/kif_to_csa.py ~/kifs/kif_data converted   # -> converted/shogiwars-*.csa
```

## 1. Build rank-bucketed data

The bands are given in **ordinals** (see the mapping above). For example
`--bands 31,33` makes three buckets: kyu (`<31`), 初段-二段 (`31-32`), 三段+ (`33+`).

```bash
# from the converted Shogi Wars CSA
python human/csa_to_hcpe_by_rating.py converted out --bands 31,33 --filter_moves 20

# or, for numeric-rated CSA (e.g. floodgate), use rating boundaries directly
python human/csa_to_hcpe_by_rating.py ~/csa out \
    --bands 1500,1800,2100,2400 --filter_moves 20 --test_ratio 0.05
```

Each position is bucketed by the **rating of the player to move**, and the
training target is the move that player actually chose. Output:

```
out/
  0000-1499/ {train,test}.hcpe
  1500-1799/ {train,test}.hcpe
  1800-2099/ {train,test}.hcpe
  2100-2399/ {train,test}.hcpe
  2400-up/   {train,test}.hcpe
```

Rating extraction requires the CSA records to carry `'black_rate:` / `'white_rate:`
comment lines (floodgate-style). Records without ratings are skipped unless you
pass `--allow_unrated` (they then go to an `unrated/` band).

## 2. Train one model per rating band

```bash
python -m pydlshogi2.train out/1800-2099/train.hcpe out/1800-2099/test.hcpe \
    --gpu 0 --val_lambda 1.0
```

Use `--val_lambda 1.0` (outcome-only value): for imitation the **policy** is
what matters. Evaluate with policy move-matching accuracy (the trainer already
logs `test accuracy`, the fraction of positions where the top policy move equals
the human move) rather than playing strength.

## 3. Play human-like

Play with little or no search so the human-trained policy is not "corrected"
toward superhuman moves — e.g. a single playout / direct policy sampling. (A
dedicated policy-only player can be added later.)
