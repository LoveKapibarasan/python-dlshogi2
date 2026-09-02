# Training Pipeline

From raw game records to a checkpoint. Source:
[`utils/csa_to_hcpe.py`](https://github.com/LoveKapibarasan/python-dlshogi2/blob/main/utils/csa_to_hcpe.py),
[`pydlshogi2/dataloader.py`](https://github.com/LoveKapibarasan/python-dlshogi2/blob/main/pydlshogi2/dataloader.py),
[`pydlshogi2/train.py`](https://github.com/LoveKapibarasan/python-dlshogi2/blob/main/pydlshogi2/train.py).

## CSA → HCPE

```bash
python utils/csa_to_hcpe.py <csa_dir> train.hcpe test.hcpe \
    --filter_moves 50 --filter_rating 3500 --test_ratio 0.1
```

HCPE (`HuffmanCodedPosAndEval`) is a **fixed-size record** holding a Huffman-coded
position, the move played, an evaluation and the game result. Fixed size is the
reason the whole pipeline is simple: `np.fromfile` loads a file into an array in
one call, and `cat a.hcpe b.hcpe > merged.hcpe` is a valid merge.

The converter drops games that would teach the wrong thing:

| Filter | Why |
|--------|-----|
| `endgame not in ('%TORYO', '%SENNICHITE', '%KACHI')` | Abandoned / illegal-move / timeout games have no meaningful result |
| `len(kif.moves) < --filter_moves` (50) | Very short games are usually disconnections |
| rating `< --filter_rating` (3500) | Weak engines teach weak moves |

The train/test split is done **by file**, not by position, with
`train_test_split` over the file list. That matters: positions from the same
game are highly correlated, so splitting positions would leak the test set into
training and make the reported accuracy meaningless.

The stored `eval` is always **from black's perspective**. Every consumer flips
it for white; see below.

### Human-like variant

`human/` is a separate sub-project that buckets the same conversion by player
strength (Maia-chess style) to imitate a rating band rather than to maximise
strength. It shares the network and feature encoding but nothing else. See
[`human/README.md`](https://github.com/LoveKapibarasan/python-dlshogi2/blob/main/human/README.md).

## The value target: `--val_lambda` and `--eval_coef`

The obvious value target is the game outcome `z ∈ {0, 0.5, 1}`. It is also very
noisy: one blunder at move 90 relabels all 90 preceding positions, including the
ones that were genuinely winning.

So the target blends the outcome with the search evaluation stored in the record:

```python
p = 1 / (1 + exp(-eval_stm / eval_coef))          # eval → win rate
target = val_lambda * z + (1 - val_lambda) * p
```

- `--val_lambda 1.0` — outcome only (the original behaviour).
- `--val_lambda 0.333` (default) — mostly the evaluation, anchored by the result.
- `--eval_coef 600` — the sigmoid temperature. It is the same 600 used by the
  engine's own `cp = -log(1/p - 1) * 600` PV output and by
  `selfplay.winrate_to_cp`, so evaluations round-trip consistently through the
  whole system. Change it in one place and you must change it in all of them.

`make_eval_winrate` flips the sign for white, so the target is always the win
probability **of the side to move** — matching the side-to-move-relative input
features and the single-output value head.

## The data loader

`HcpeDataLoader` keeps the whole dataset in memory as one numpy array
(`np.concatenate` of every file) and builds batches into **pre-allocated pinned
tensors**, which makes the host→device copy asynchronous-friendly. A single
background thread (`ThreadPoolExecutor(max_workers=1)`) prepares the next batch
while the GPU works on the current one — the feature construction is a Python
loop over `cshogi`, so overlapping it matters.

Two consequences worth knowing:

- Memory is proportional to the dataset. A very large HCPE corpus needs a
  machine that can hold it.
- The last partial batch of an epoch is dropped (`pre_fetch` returns early when
  fewer than `batch_size` records remain).

## Training

```bash
python -m pydlshogi2.train train.hcpe test.hcpe \
    --gpu 0 --epoch 10 --batchsize 1024 --lr 0.01 \
    --checkpoint checkpoints/checkpoint-{epoch:03}.pth \
    --metrics metrics/train-sl.jsonl
```

- **Loss** = `CrossEntropyLoss(policy)` + `BCEWithLogitsLoss(value)`, unweighted.
- **Optimiser** = SGD, momentum 0.9, weight decay 1e-4. `--lr` always wins,
  including on resume — the optimiser state is restored but its learning rate is
  overwritten, so you can lower the LR when continuing.
- `--amp` enables bfloat16 autocast (CUDA only), `--compile` wraps the model in
  `torch.compile`. Because `torch.compile` renames state-dict keys, the
  **uncompiled** module is kept as `base_model` and is what gets saved.
- Evaluation happens twice: every `--eval_interval` steps on a single random
  test mini-batch (cheap, noisy — good for curves), and at the end of every
  epoch over the entire test set (slow, trustworthy — good for comparing runs).
  In the metrics these are `scope: "interval"` and `scope: "epoch"`; see
  [Metrics and Dashboard](Metrics-and-Dashboard).

## Preemption safety

Spot/preemptible GPU instances can disappear at any moment, so training is built
to survive it:

- `SIGTERM` / `SIGINT` set a flag; the current step finishes, a checkpoint is
  written, and the process exits `0`. The handler is registered **before** data
  loading, so a signal arriving during the (slow) load is not lost.
- `--save_interval N` writes a checkpoint every N steps as well, so at most N
  steps of work is ever at risk.
- The checkpoint carries `epoch`, `t` (global step), model, optimiser state and
  the architecture — everything needed to continue.

```bash
# initial run: a single rolling checkpoint
python -m pydlshogi2.train train.hcpe test.hcpe --gpu 0 --epoch 10 \
    --save_interval 1000 --checkpoint checkpoints/latest.pth \
    --metrics metrics/train-sl.jsonl --run_id sl-2026-09

# after a preemption: same paths, same run id
python -m pydlshogi2.train train.hcpe test.hcpe --gpu 0 --epoch 10 \
    --resume checkpoints/latest.pth --checkpoint checkpoints/latest.pth \
    --metrics metrics/train-sl.jsonl --run_id sl-2026-09
```

`--epoch` on a resume is the number of **additional** epochs, not a target.

Passing the same `--run_id` keeps the metrics of the original run and its
resumes as one logical run, so the dashboard draws one continuous curve.

---

See also: [Architecture](Architecture),
[Reinforcement Learning](Reinforcement-Learning),
[Metrics and Dashboard](Metrics-and-Dashboard)
