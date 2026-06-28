# Human-like shogi AI (separate sub-project)

This directory is a **separate effort** from the strong self-play engine. The
goal here is *human-likeness*, not maximal strength: imitate how humans of a
given skill level play, in the spirit of [Maia Chess](https://maiachess.com/).

It does **not** touch the reinforcement-learning pipeline. It only reuses the
shared network (`pydlshogi2.network`) and feature encoding (`pydlshogi2.features`)
at training time; the data tooling here depends only on `cshogi`.

## 1. Build rating-bucketed data from human game records

```bash
python human/csa_to_hcpe_by_rating.py ~/kif out \
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
