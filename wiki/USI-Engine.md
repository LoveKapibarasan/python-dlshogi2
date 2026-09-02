# USI Engine

The engine speaks USI (Universal Shogi Interface) and works with any USI GUI —
ShogiGUI, Shogidokoro, and others.

## Two backends

```bash
python -m pydlshogi2.player.mcts_player     # PyTorch,  ./mcts_player.sh
python -m pydlshogi2.player.onnx_player     # ONNX,     ./onnx_player.sh
```

| | PyTorch (`mcts_player`) | ONNX (`onnx_player`) |
|---|---|---|
| Model | `.pth` checkpoint (~57 MB) | `.onnx` (~29 MB) |
| Contains | weights **+ optimiser state** | weights only |
| Needs | torch | onnxruntime |
| Use for | resuming training, quick tests | playing |

The size difference is not compression — a `.pth` stores the SGD state so
training can resume, which a player never needs.

The ONNX player subclasses the PyTorch one and replaces only `infer()`, so the
search, mate handling, time management and USI options are literally the same
code. Anything on the [MCTS](MCTS) page applies to both.

Export before using the ONNX player:

```bash
python utils/export_onnx.py checkpoints/checkpoint.pth model/model.onnx
```

## Registering in a GUI

1. Point the GUI at `mcts_player.sh` / `onnx_player.sh` (or the `.bat` on
   Windows) rather than at the Python module — the wrapper scripts pick the
   interpreter.
2. Set `modelfile` in the engine options.
3. Set `gpu_id` to `-1` if there is no CUDA GPU.

Registration fails most often because the GUI launched the engine with a
different working directory or a different Python than the shell you tested in.
`_resolve_model_path` handles the relative-path half of that; a wrong
interpreter shows up as an import error in the GUI's engine log.

## Options worth changing

Full table on the [MCTS](MCTS#time-management) page. In practice:

- **`batchsize`** — the throughput knob. Larger batches use the GPU better but
  make each playout's information staler within a batch. 32 is the default.
- **`gpu_id -1`** — CPU play. Lower `batchsize` accordingly.
- **`resign_threshold`** — percent. `0` never resigns, which is what you want
  when generating games for analysis.
- **`mate_root_ply`** — a deeper one-shot root mate search finds more forced
  wins at a fixed cost per move.
- **`pv_interval 0`** — silence `info pv` output.
- **`USI_Ponder`** — think on the opponent's clock. The search runs unbounded
  until `stop` or `ponderhit`.

Spin options representing fractions are integers ×100: `c_puct 100` means 1.00.

## Pre-trained models

`checkpoints/checkpoint.pth` and `checkpoints/checkpoint-001.pth` ship with the
repository (Floodgate 2020, rating ≥ 3500, ≥ 50 moves).

The bundled `model/model-0000167.onnx` and `model/model-0000225kai.onnx` are
**dlshogi-format** models with a two-input (`input1`/`input2`) graph and are
**not** compatible with this player — export your own from a `.pth`. See
[Troubleshooting](Troubleshooting).

---

See also: [MCTS](MCTS), [Architecture](Architecture)
