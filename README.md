# python-dlshogi2

Deep learning shogi AI engine using a policy-value network and Monte Carlo Tree Search (MCTS), inspired by the AlphaGo Zero approach.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LoveKapibarasan/python-dlshogi2/blob/main/notebooks/train.ipynb)

---

## Overview

python-dlshogi2 trains a ResNet-based neural network on shogi game records, then uses the trained model to play shogi via MCTS. It communicates with shogi GUIs using the **USI (Universal Shogi Interface)** protocol.

**Key features:**
- Configurable policy-value ResNet with optional Squeeze-and-Excitation (default: 20 blocks, 256 channels, SE)
- Supervised training with an evaluation-blended value target
- Self-play reinforcement learning loop (AlphaZero / expert-iteration style)
- MCTS with virtual loss, FPU and an AlphaZero-style PUCT term
- Mate detection (configurable root mate search) and draw recognition
- Ponder support (thinking during opponent's turn)
- PyTorch and ONNX (CUDA/TensorRT) inference backends
- Resignation based on configurable win-rate threshold
- Sphinx API documentation generated from in-source docstrings

---

## Requirements

- Python 3.x
- PyTorch
- onnxruntime (for ONNX player)
- numpy
- cshogi
- scikit-learn (for data conversion utilities)

---

## Installation

```bash
pip install cshogi
pip install git+https://github.com/LoveKapibarasan/python-dlshogi2.git
```

---

## Google Colab

You can train and evaluate the model using Google Colab without any local GPU setup.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LoveKapibarasan/python-dlshogi2/blob/main/notebooks/train.ipynb)

Install the package in a Colab cell:

```python
!pip install cshogi
!pip install git+https://github.com/LoveKapibarasan/python-dlshogi2.git
```

Then run training:

```python
!python -m pydlshogi2.train train.hcpe test.hcpe \
    --gpu 0 \
    --epoch 10 \
    --batchsize 1024 \
    --checkpoint checkpoints/checkpoint-{epoch:03}.pth
```

---

## Data Preparation

Convert CSA game records to HCPE training data:

```bash
python utils/csa_to_hcpe.py <csa_dir> train.hcpe test.hcpe \
    --filter_moves 50 \
    --filter_rating 3500 \
    --test_ratio 0.1
```

| Option | Default | Description |
|--------|---------|-------------|
| `--filter_moves` | 50 | Minimum number of moves per game |
| `--filter_rating` | 3500 | Minimum player rating |
| `--test_ratio` | 0.1 | Fraction of data used for testing |

---

## Training

```bash
python -m pydlshogi2.train train.hcpe test.hcpe \
    --gpu 0 \
    --epoch 10 \
    --batchsize 1024 \
    --lr 0.01 \
    --checkpoint checkpoints/checkpoint-{epoch:03}.pth
```

| Option | Default | Description |
|--------|---------|-------------|
| `--gpu` | 0 | GPU ID (-1 for CPU) |
| `--epoch` | 10 | Number of training epochs |
| `--batchsize` | 1024 | Mini-batch size |
| `--lr` | 0.01 | Learning rate (SGD with momentum) |
| `--checkpoint` | — | Checkpoint path template |
| `--blocks` / `--channels` / `--fcl` | 20 / 256 / 256 | Network size (ignored when `--resume`) |
| `--no_se` | off | Disable Squeeze-and-Excitation blocks |
| `--val_lambda` | 0.333 | Weight on game outcome vs. search eval in the value target |
| `--eval_coef` | 600 | Sigmoid temperature for eval→win-rate |
| `--amp` | off | bfloat16 autocast (mixed precision) |
| `--compile` | off | Wrap model with `torch.compile` |
| `--save_interval` | 0 | Save a checkpoint every N steps (0 = epoch end only) |
| `--resume` | — | Resume from a checkpoint (model + optimizer + step + architecture) |

### Interrupting & resuming (preemptible instances)

Training is preemption-safe: on `SIGTERM`/`SIGINT` it checkpoints after the
current step and exits cleanly, and `--save_interval` writes periodic
checkpoints mid-epoch. Resume with the **same checkpoint path**:

```bash
# initial run (writes a single rolling checkpoint)
python -m pydlshogi2.train train.hcpe test.hcpe --gpu 0 --epoch 10 \
    --save_interval 1000 --checkpoint checkpoints/latest.pth

# after a preemption, continue from where it stopped
python -m pydlshogi2.train train.hcpe test.hcpe --gpu 0 --epoch 10 \
    --resume checkpoints/latest.pth --checkpoint checkpoints/latest.pth
```

`--epoch` is the number of *additional* epochs to run on resume. The optimizer
state, global step and network architecture are all restored from the
checkpoint.

---

## Playing (USI Engine)

### PyTorch player

```bash
python -m pydlshogi2.player.mcts_player
# or
./mcts_player.sh
```

### ONNX player

The ONNX backend now uses the **same feature representation as the PyTorch
backend**, so it plays models exported from this repository's own checkpoints.
First export a trained checkpoint to ONNX:

```bash
python utils/export_onnx.py checkpoints/checkpoint.pth model/model.onnx
```

Then run the engine:

```bash
python -m pydlshogi2.player.onnx_player
# or
./onnx_player.sh
```

> **Note:** the previously shipped `model/model-0000167.onnx` and
> `model/model-0000225kai.onnx` are *dlshogi-format* models (two-input
> `input1`/`input2` graph) and are **no longer compatible** with this player.
> Export your own model from a `.pth` checkpoint as shown above.

### USI options

| Option | Default | Description |
|--------|---------|-------------|
| `modelfile` | — | Path to checkpoint or ONNX model |
| `gpu_id` | 0 | GPU ID (-1 for CPU) |
| `batchsize` | 32 | Neural network batch size |
| `resign_threshold` | 5 | Win rate (%) below which to resign |
| `c_puct` | 1.0 | MCTS exploration constant |
| `temperature` | 1.0 | Policy softmax temperature |
| `time_margin` | 1000 | Time margin in milliseconds |
| `byoyomi_margin` | 100 | Byoyomi margin in milliseconds |
| `pv_interval` | 500 | PV info output interval (ms) |
| `debug` | false | Enable debug output |

Connect the engine to any USI-compatible shogi GUI (e.g., Shogidokoro, ShogiGUI).

---

## Model Architecture

- **Input:** 104 feature planes on a 9×9 board (piece positions + captured pieces for both sides; `FEATURES_NUM`)
- **Backbone:** Convolutional layer → ResNet blocks with optional Squeeze-and-Excitation (default 20 blocks × 256 channels, batch norm, ReLU)
- **Policy head:** 2,187 move outputs (27 planes × 81 squares = 20 directions + 7 drop pieces; `MOVE_LABELS_NUM`)
- **Value head:** Single sigmoid output (estimated win probability)

The architecture (`--blocks`, `--channels`, `--fcl`, `--no_se`) is configurable
and embedded in each checkpoint, so players and the ONNX exporter reconstruct it
automatically. Checkpoints saved before this feature load as a legacy 10×192
SE-free network.

## Self-play Reinforcement Learning

Generate self-play games and improve the model in a loop:

```bash
# one batch of self-play games (single process)
python -m pydlshogi2.selfplay checkpoints/checkpoint.pth selfplay.hcpe \
    --games 1000 --playouts 800 --gpu 0

# parallel self-play (recommended): a single process is CPU-bound on the MCTS
# tree and leaves the GPU idle; N workers sharing the GPU multiply throughput.
WORKERS=8 GAMES=1000 PLAYOUTS=400 GPU=0 \
    ./selfplay_parallel.sh checkpoints/checkpoint.pth selfplay.hcpe

# full generate -> train -> promote loop (uses parallel self-play internally)
WORKERS=8 ./rl_loop.sh checkpoints/checkpoint.pth
```

The RL loop honours `WORKERS`, `GAMES`, `PLAYOUTS`, `SELFPLAY_BATCHSIZE`,
`ITERATIONS`, `EPOCHS`, `LR`, `VAL_LAMBDA`, `GPU` and `WORKDIR`.

## Running on a GPU server (Vast.ai)

One-shot bootstrap + background jobs on a fresh CUDA instance:

```bash
# 1) provision: venv + CUDA torch + deps + onnxruntime-gpu + editable install
./vast_setup.sh

# 2) launch a job in the background (auto-runs setup if needed).
#    Logs to logs/<mode>-<timestamp>.log; survives SSH disconnects.
./vast_run.sh train    train.hcpe test.hcpe --gpu 0 --amp        # supervised
./vast_run.sh selfplay checkpoints/checkpoint.pth sp.hcpe --games 1000 --gpu 0
./vast_run.sh rl       checkpoints/checkpoint.pth                # full self-play loop
```

Follow a job with `tail -f logs/<mode>-*.log` and stop it with
`kill $(cat logs/<mode>.pid)`. The RL loop is tunable via environment variables
(`ITERATIONS`, `GAMES`, `PLAYOUTS`, `EPOCHS`, `LR`, `VAL_LAMBDA`, `GPU`,
`WORKDIR`).

## Documentation

API docs are generated from in-source reStructuredText docstrings with Sphinx:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html   # or: cd docs && make html
```

---

## Pre-trained Models

The repository ships with pre-trained weights so you can try the engine immediately without training.

| File | Format | Size | Epochs | Steps |
|------|--------|------|--------|-------|
| `checkpoints/checkpoint-001.pth` | PyTorch | 57 MB | 1 | 2,495 |
| `checkpoints/checkpoint.pth` | PyTorch | 57 MB | 3 | 195,555 |
| `model/model-0000167.onnx` | ONNX | 29 MB | — | — |
| `model/model-0000225kai.onnx` | ONNX | 29 MB | — | — |

All models were trained on **Floodgate 2020** game records (rating ≥ 3500, ≥ 50 moves).

**PyTorch vs ONNX:**
- `.pth` files are ~57 MB because they store both the model weights and the optimizer state (needed to resume training).
- `.onnx` files are ~29 MB because they contain only the model weights — half the size, faster to load, and usable without PyTorch.
- Use `.pth` if you want to resume training; use `.onnx` for playing.

**`checkpoints/.gitignore` and `model/.gitignore`** both contain `*`, which prevents any newly generated files from being accidentally committed, while leaving the pre-trained files above tracked by git.

---

## Training Outputs

After running `train.py`, the following files are generated:

```
checkpoints/
  checkpoint-001.pth    # saved after epoch 1  (~57 MB)
  checkpoint-002.pth    # saved after epoch 2  (~57 MB)
  ...

model/
  *.onnx                # only if you run the ONNX export step (~29 MB each)
```

### Checkpoint file structure

Each `.pth` file is a Python dict saved with `torch.save`:

```python
{
    'epoch':     1,      # which epoch this was saved at
    't':         2495,   # total training steps completed
    'model':     {...},  # network weights (138 tensors, ResNet 10-block 192ch)
    'optimizer': {...},  # SGD state — required only to resume training
}
```

To load a checkpoint for inference only (no optimizer needed):

```python
import torch
from pydlshogi2.network.policy_value_resnet import PolicyValueNetwork

model = PolicyValueNetwork()
ckpt = torch.load('checkpoints/checkpoint.pth', map_location='cpu')
model.load_state_dict(ckpt['model'])
model.eval()
```

To resume training from a checkpoint, pass it via `--resume`:

```bash
python -m pydlshogi2.train train.hcpe test.hcpe \
    --resume checkpoints/checkpoint.pth \
    --epoch 10
```

---

## License

See [LICENSE](LICENSE).
