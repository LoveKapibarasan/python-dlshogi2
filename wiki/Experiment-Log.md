# Experiment Log

Running record of what was trained, with what settings, and what came out.
Add a section per experiment, newest first. Keep it short — the numbers live in
the metrics files; this page is for the **intent** and the **conclusion**, which
no automated record captures.

## Template

```markdown
## YYYY-MM-DD — one-line summary

- **run_id**: `20260902-101500-1a2b3c4d`
- **commit**: `fc71d53`
- **hardware**: RTX 4090 (Vast.ai)
- **data**: Floodgate 2020, rating ≥ 3500, ≥ 50 moves — 12.0M / 1.3M positions
- **command**:
  ```bash
  python -m pydlshogi2.train train.hcpe test.hcpe --gpu 0 --epoch 10 \
      --batchsize 1024 --lr 0.01 --val_lambda 0.333 --amp \
      --checkpoint checkpoints/checkpoint-{epoch:03}.pth \
      --metrics metrics/train-sl.jsonl
  ```
- **question**: what was this run trying to find out?
- **result**: test policy accuracy 0.412, value accuracy 0.712 (epoch 10)
- **conclusion**: what you now believe, and what to try next.
```

Most of those fields are already in the `run` record of the metrics file — the
`run_id` alone is enough to recover the command, commit, host and GPU. Copy them
in anyway: this page should stay readable without tooling.

## Recording a run

```bash
python -m pydlshogi2.train ... --metrics metrics/train-sl.jsonl
streamlit run dashboard/app.py     # Runs tab → select the run → full arguments
```

See [Metrics and Dashboard](Metrics-and-Dashboard).

## Conventions

- **One section per experiment**, not per process. A run preempted and resumed
  three times is one experiment — pass the same `--run_id` and it will be one
  run in the dashboard too.
- **Record the failures.** A hyper-parameter that made things worse is worth
  more here than another confirmation that the defaults work.
- **Note `git_dirty`.** A run recorded with uncommitted changes cannot be
  reproduced from its commit; say what was modified.
- **Link the checkpoint** that came out of it, so the model files are traceable
  back to a decision.

---

## Baseline (shipped checkpoints)

The checkpoints in the repository predate this log; they are recorded here so
later runs have something to compare against.

- **data**: Floodgate 2020, rating ≥ 3500, ≥ 50 moves
- **`checkpoints/checkpoint-001.pth`** — epoch 1, 2,495 steps
- **`checkpoints/checkpoint.pth`** — epoch 3, 195,555 steps
- **architecture**: no embedded config → loaded as the legacy `10 × 192`, SE off
- **metrics**: none (predates the metrics writer)

---

See also: [Metrics and Dashboard](Metrics-and-Dashboard),
[Training Pipeline](Training-Pipeline),
[Improvement Backlog](Improvement-Backlog),
[Evaluation and Rating](Evaluation-and-Rating)
