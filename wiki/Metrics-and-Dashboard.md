# Metrics and Dashboard

Training and self-play write a JSON Lines record stream alongside their text
logs, and a Streamlit dashboard reads it back as a browsable development
history. Source:
[`pydlshogi2/metrics.py`](https://github.com/LoveKapibarasan/python-dlshogi2/blob/main/pydlshogi2/metrics.py),
[`dashboard/`](https://github.com/LoveKapibarasan/python-dlshogi2/tree/main/dashboard).

## Why JSONL rather than TensorBoard

TensorBoard is the obvious choice for scalar curves, and it would work. It was
not chosen because the questions being asked here are mostly *not* about curves:

- "Which commit and which hyper-parameters produced `checkpoint-007.pth`?"
- "How many games per hour did 8 workers manage in iteration 3?"
- "Which of these six runs was interrupted, and at what step?"

Those want a queryable record of runs, not a scalar time series. A flat JSONL
file answers them with `json.loads` and no server, survives being `scp`-ed off a
dead spot instance, and diffs sensibly. Nothing prevents adding a
`SummaryWriter` later for the curves specifically.

## Recording

```bash
python -m pydlshogi2.train train.hcpe test.hcpe --gpu 0 --epoch 10 \
    --checkpoint checkpoints/checkpoint-{epoch:03}.pth \
    --metrics metrics/train-sl.jsonl

python -m pydlshogi2.selfplay checkpoints/checkpoint.pth selfplay.hcpe \
    --games 1000 --metrics metrics/selfplay.jsonl

WORKERS=8 ./rl_loop.sh checkpoints/checkpoint.pth   # writes rl/metrics/ automatically
```

Without `--metrics` the writer is a no-op, so nothing changes for anyone who
does not want it.

Put the metrics next to the checkpoints. Retrieving a remote run then means the
same `scp` you already do for the weights, and the two cannot drift apart.

## Schema

One file per run. The first record describes the run; the rest are samples and
events. Every record is flushed as written, so a preempted run still leaves a
readable history.

```json
{"type": "run", "run_id": "20260902-101500-1a2b3c4d", "kind": "train",
 "git_commit": "fc71d53...", "git_dirty": false, "hostname": "vast-1",
 "gpu_name": "RTX 4090", "args": {"lr": 0.01, "batchsize": 1024}}
{"type": "event", "event": "data_loaded", "train_positions": 12000000, "network": {"blocks": 20}}
{"type": "metric", "scope": "interval", "epoch": 1, "step": 100, "train_loss_total": 4.31}
{"type": "event", "event": "checkpoint", "path": "checkpoints/checkpoint-001.pth"}
{"type": "event", "event": "run_end", "status": "completed"}
```

`kind` is `train`, `selfplay` or `rl`. `scope` separates granularities:

| `scope` | Written by | Meaning |
|---------|-----------|---------|
| `interval` | train | every `--eval_interval` steps, one random test mini-batch |
| `epoch` | train | end of epoch, the full test set |
| `game` | selfplay | one game: plies, result, seconds |
| `summary` | selfplay | one worker's totals: games, positions, win/draw split, games/hour |
| `iteration` | rl_loop.sh | one RL iteration: model, data, checkpoint, seconds |

`git_dirty` is worth reading before trusting a comparison: a run recorded with
uncommitted changes cannot be reproduced from its commit alone.

## Run ids and resumes

A run id is `20260902-101500-1a2b3c4d` — a sortable timestamp plus a random
suffix, so runs started in the same second on different workers stay distinct.

By default each process gets a fresh id, so **a preempted run and its resume
appear as two runs**, the second carrying the `resume` path in its arguments.
That is the honest default: they really were two processes, possibly on
different hardware or a different commit.

Pass `--run_id` when you would rather see one continuous run. Since the step
counter is restored from the checkpoint, the samples line up on the step axis
either way — the only difference is grouping.

`rl_loop.sh` uses this: the loop is `rl-<timestamp>` and each iteration's
training run is `<loop-id>-train-00i`, which is how the dashboard knows they
belong together. Set `RL_RUN_ID` to continue an interrupted loop under its
original id.

## The dashboard

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

Set the directories in the sidebar, or via `DLSHOGI_METRICS_DIR` /
`DLSHOGI_CHECKPOINT_DIR`. The metrics directory is scanned **recursively** for
`*.jsonl`, so a directory of runs downloaded from several machines works without
preprocessing.

| Tab | Shows |
|-----|-------|
| Runs | every run: start time, commit, host/GPU, hyper-parameters, last step, final accuracy; select one for its full argument list |
| 学習曲線 | several runs' losses and accuracies overlaid on a step axis, switchable between `interval` and `epoch` |
| RL ループ | self-play per iteration with workers summed — games, win/draw split, mean game length — and each iteration's artifacts |
| チェックポイント | model files with size and mtime, and on request the architecture embedded in a `.pth` |

Dashboard dependencies live in `dashboard/requirements.txt`, separate from the
engine's, so training and playing never have to install Streamlit.

## Reading the metrics without the dashboard

`dashboard/metrics_store.py` is standard-library only:

```python
import sys; sys.path.insert(0, 'dashboard')
import metrics_store

records = metrics_store.load('rl/metrics')
for run in metrics_store.summarize_runs(records):
    print(run['run_id'], run['kind'], run['status'], run.get('last_step'))

for row in metrics_store.selfplay_by_iteration(records):
    print(row['iteration'], row['games'], row['black_win_rate'])
```

Shell scripts can append a record with no Python of their own:

```bash
python -m pydlshogi2.metrics rl/metrics/rl.jsonl \
    --type event --event iteration_end --run-id rl-20260902 \
    --set iteration=3 --set checkpoint=rl/checkpoint-003.pth
```

`--set KEY=VALUE` decodes the value as JSON when it parses (so `3` is a number)
and keeps it as a string otherwise (so a path stays a path).

## Tests

```bash
python -m unittest discover -s tests
```

`tests/test_metrics.py` covers the writer and the loader and needs neither torch
nor cshogi, so it runs anywhere — including in CI.

---

See also: [Training Pipeline](Training-Pipeline),
[Reinforcement Learning](Reinforcement-Learning),
[Experiment Log](Experiment-Log)
