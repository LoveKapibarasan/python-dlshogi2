# Reinforcement Learning

Expert-iteration / AlphaZero-style loop: play games with the current model,
train on them, promote the result, repeat. Source:
[`pydlshogi2/selfplay.py`](https://github.com/LoveKapibarasan/python-dlshogi2/blob/main/pydlshogi2/selfplay.py),
[`selfplay_parallel.sh`](https://github.com/LoveKapibarasan/python-dlshogi2/blob/main/selfplay_parallel.sh),
[`rl_loop.sh`](https://github.com/LoveKapibarasan/python-dlshogi2/blob/main/rl_loop.sh).

```bash
WORKERS=8 ./rl_loop.sh checkpoints/checkpoint.pth
```

**Start from a supervised checkpoint.** Self-play from random weights would
spend an enormous amount of compute rediscovering that pieces move.

## What one iteration does

```
iteration i
 ├─ 1. selfplay_parallel.sh  current model → rl/selfplay-00i.hcpe
 ├─ 2. train --resume current, on ALL selfplay data so far → rl/checkpoint-00i.pth
 └─ 3. promote: current := rl/checkpoint-00i.pth
```

Step 2 trains on `rl/selfplay-*.hcpe` — every iteration generated so far, not
just the newest. Training only on the freshest games makes the network chase its
own most recent quirks; keeping the history is a poor man's replay buffer.

The "test" file passed to the trainer is the newest batch, which also appears in
training. The resulting test numbers are therefore **a sanity check, not a
generalisation measure** — they tell you the run has not diverged, nothing more.
Judge actual progress by playing the checkpoints against each other.

## Resumability

Both artifacts of an iteration are checked before doing the work:

```bash
if [ -s "$NEXT" ]; then ... skipping; fi     # already trained
if [ -s "$DATA" ]; then ... reusing; fi      # already generated
```

So re-running `rl_loop.sh` after a crash or preemption picks up where it
stopped instead of regenerating games. Note the check is `-s` (non-empty), not a
completeness check — a truncated `.hcpe` from a process killed mid-write will be
reused as-is. Delete a suspect file to force regeneration.

## Self-play details

Each position is recorded as:

| Field | Value |
|-------|-------|
| `hcp` | the position before the move |
| `bestMove16` | the **most-visited** move (`argmax` of visit counts) |
| `eval` | the MCTS root win rate as centipawns, black's perspective |
| `gameResult` | filled in once the game ends |

The move actually **played** is not the move recorded. Moves are sampled from
the visit counts with a temperature up to `--temp_cutoff` (default ply 30) and
greedy afterwards, and Dirichlet noise (`--dirichlet_alpha 0.15`,
`--noise_eps 0.25`) is mixed into the root prior. Sampling and noise keep the
games diverse — without them every game from a deterministic engine would be
nearly identical and carry almost no information. Recording the greedy move
instead of the sampled one gives the policy head a cleaner target: "what the
search concluded", not "what the dice chose".

Recording the root win rate as `eval` is what makes `--val_lambda 0.5` possible
in the loop: the value target becomes half the eventual result, half the search's
own bootstrapped estimate.

Terminal handling covers mate, nyugyoku, all three repetition outcomes
(`REPETITION_DRAW` / `WIN` / `LOSE`) and a `--max_moves` cap (512) that declares
a draw — otherwise a weak network can shuffle forever.

## Parallel self-play

A single self-play process is CPU-bound on the Python MCTS tree and leaves the
GPU mostly idle, so `selfplay_parallel.sh` runs `WORKERS` processes that share
the GPU:

```bash
WORKERS=8 GAMES=1000 PLAYOUTS=400 GPU=0 \
    ./selfplay_parallel.sh checkpoints/checkpoint.pth selfplay.hcpe
```

Each worker gets `--seed w`, which is essential — without distinct seeds the
Dirichlet noise and the move sampling would be identical across workers and they
would produce eight copies of the same games. Outputs are merged with `cat`,
valid because HCPE records are fixed size.

Self-play data is deliberately **not** de-duplicated (unlike `utils/hcpe_dedup.py`
for human games): a repeated position with a different outcome is a legitimate
value-learning sample.

Throughput scales roughly linearly with workers until the GPU or the CPU cores
saturate. Raise `WORKERS` until games/hour stops improving — the dashboard's RL
tab reports it.

## Tunables

| Variable | Default | Notes |
|----------|--------:|-------|
| `ITERATIONS` | 20 | self-play → train cycles |
| `GAMES` | 1000 | games per iteration, across all workers |
| `PLAYOUTS` | 400 | per move; the main quality/throughput knob |
| `WORKERS` | 8 | parallel self-play processes |
| `SELFPLAY_BATCHSIZE` | 32 | inference batch per worker |
| `EPOCHS` | 1 | training epochs per iteration |
| `BATCHSIZE` | 1024 | training batch |
| `LR` | 0.002 | lower than SL — this is fine-tuning |
| `VAL_LAMBDA` | 0.5 | blend result with bootstrapped value |
| `GPU` | 0 | |
| `WORKDIR` | `rl` | where iteration artifacts land |
| `METRICS_DIR` | `$WORKDIR/metrics` | dashboard metrics |
| `RL_RUN_ID` | `rl-<timestamp>` | set it to continue an interrupted loop's history |
| `PYTHON` | `python` | set to `.venv/bin/python` on Vast.ai |

`PLAYOUTS` and `GAMES` trade against each other for a fixed compute budget:
more playouts means better targets per position, more games means more
positions. 400 playouts is well below AlphaZero's 800; raise it if the policy
stops improving.

## What gets recorded

`rl_loop.sh` needs no extra flags to produce dashboard metrics — it writes to
`$WORKDIR/metrics`:

```
rl/metrics/
  rl.jsonl                 # the loop: one record per iteration, with timings
  selfplay-001-w0.jsonl    # per-worker games, results, throughput
  train-001.jsonl          # the training run inside iteration 1
```

See [Metrics and Dashboard](Metrics-and-Dashboard).

---

See also: [MCTS](MCTS), [Training Pipeline](Training-Pipeline),
[Environments](Environments)
