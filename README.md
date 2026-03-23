# python-dlshogi2

Deep learning shogi AI engine using a policy-value network and Monte Carlo Tree Search (MCTS), inspired by the AlphaGo Zero approach.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LoveKapibarasan/python-dlshogi2/blob/main/notebooks/train.ipynb)

---

## Overview

python-dlshogi2 trains a ResNet-based neural network on shogi game records, then uses the trained model to play shogi via MCTS. It communicates with shogi GUIs using the **USI (Universal Shogi Interface)** protocol.

**Key features:**
- Policy-value ResNet (10 blocks, 192 channels)
- MCTS with virtual loss for parallelism and UCB-based exploration
- Mate detection and draw recognition
- Ponder support (thinking during opponent's turn)
- PyTorch and ONNX inference backends
- Resignation based on configurable win-rate threshold

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

---

## Playing (USI Engine)

### PyTorch player

```bash
python -m pydlshogi2.player.mcts_player
# or
./mcts_player.sh
```

### ONNX player

```bash
python -m pydlshogi2.player.onnx_player
# or
./onnx_player.sh
```

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

- **Input:** 38 feature planes on a 9×9 board (piece positions + captured pieces for both sides)
- **Backbone:** Convolutional layer → 10 ResNet blocks (192 channels, batch norm, ReLU)
- **Policy head:** 1,629 move outputs (20 directions × 81 squares + hand drops)
- **Value head:** Single sigmoid output (estimated win probability)

---

## Pre-trained Models

| File | Format | Training data |
|------|--------|---------------|
| `checkpoints/checkpoint.pth` | PyTorch | Floodgate 2020 (full year) |
| `model/model-0000167.onnx` | ONNX | Floodgate 2020 (full year) |
| `model/model-0000225kai.onnx` | ONNX | Floodgate 2020 (full year) |

---

## License

See [LICENSE](LICENSE).
