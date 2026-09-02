# Troubleshooting

Known pitfalls, in roughly the order people hit them.

## The bundled `.onnx` models do not load

`model/model-0000167.onnx` and `model/model-0000225kai.onnx` are **dlshogi-format**
models: a two-input graph (`input1` / `input2`) with a different feature
encoding. `OnnxPlayer` calls

```python
self.session.run(["output_policy", "output_value"], {"input": x})
```

so it fails on them by name. They are kept only for reference. Export your own:

```bash
python utils/export_onnx.py checkpoints/checkpoint.pth model/model.onnx
```

The exporter bakes the `sigmoid` into the value output and marks the batch axis
dynamic, so one exported file serves any `batchsize`.

## An old checkpoint loads as a 10×192 network

Checkpoints saved before the architecture was embedded have no `'network'` key,
and `load_network` falls back to `LEGACY_NETWORK_CONFIG` = `10 × 192`, SE off.
If the checkpoint really was a different shape, `load_state_dict` raises a size
mismatch. There is no way to recover the shape from the weights automatically —
rebuild the config by hand and load with an explicit `build_network(...)`.

## `--blocks` / `--channels` seem to be ignored

They are, whenever `--resume` is given. The checkpoint's own architecture wins,
because its weights could not be loaded into a different shape. To change the
architecture you start a new run without `--resume`.

## Training dies with an OOM before the first step

`HcpeDataLoader` loads **the entire dataset into RAM** (`np.concatenate` of every
file). That is host memory, not GPU memory, and it is proportional to the corpus.
Split the data and train on fewer files per run, or use a bigger machine.

If it is GPU memory instead, lower `--batchsize`, or add `--amp`.

## The last few positions of each epoch are never trained on

By design: `pre_fetch` returns early when fewer than `batch_size` records remain,
so a partial final batch is dropped. With shuffling on, a different remainder is
dropped each epoch, so nothing is systematically excluded.

## Self-play workers produce identical games

Each worker needs a distinct `--seed`; `selfplay_parallel.sh` passes `--seed w`.
Running several `pydlshogi2.selfplay` processes by hand without seeds gives you
N copies of the same games — the Dirichlet noise and the move sampling are the
only sources of variety, and both come from the same default-seeded RNG.

## `rl_loop.sh` skips an iteration that did not finish

The resume check is `[ -s "$FILE" ]` — non-empty, not complete. A `.hcpe`
truncated by a process killed mid-write will be reused as-is, and a partially
written checkpoint will be treated as trained. Delete the suspect file to force
the iteration to redo it.

## RL test accuracy looks great and the engine is not stronger

The RL loop passes the newest self-play batch as **both** training and test
data. Those numbers are a divergence check, not a generalisation measure. Judge
progress by playing checkpoints against each other.

## Training was preempted and the metrics show two runs

That is the default: each process gets a fresh `run_id`. Pass the same
`--run_id` to both invocations (and `RL_RUN_ID` to `rl_loop.sh`) if you want one
continuous run in the dashboard. The step axis lines up either way, because the
step counter is restored from the checkpoint.

## The dashboard shows nothing

- Metrics only exist if `--metrics` was passed. Training without it is silent by
  design.
- The metrics directory is scanned recursively for `*.jsonl` — check the
  sidebar's file count.
- The cache keys on file sizes and mtimes; if a file changed in place without
  changing either, hit **再読み込み**.

## The GUI cannot start the engine

Point the GUI at `mcts_player.sh` / `onnx_player.sh` (or the `.bat`), not at the
Python module — the wrapper picks the interpreter. Almost every failure here is
the GUI launching a different Python, or a different working directory, than the
shell where it was tested. Check the GUI's engine log for the import error.

## `torch.compile` changed my state-dict keys

It does — which is why `train.py` keeps the uncompiled module as `base_model`
and saves that. If you save a compiled model yourself, expect an `_orig_mod.`
prefix on every key.

## Evaluation values look wrong after changing `--eval_coef`

`600` appears in four places that must agree: the training target
(`make_eval_winrate`), the engine's PV output, `selfplay.winrate_to_cp`, and any
stored `eval` in existing HCPE data. Changing it for training while old data was
written with the old constant silently mislabels the value targets.

---

See also: [Training Pipeline](Training-Pipeline), [USI Engine](USI-Engine),
[Environments](Environments)
