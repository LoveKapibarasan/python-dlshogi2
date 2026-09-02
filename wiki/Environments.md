# Environments

Three ways this project gets run, and what is different about each.

## Local GPU

```bash
pip install cshogi
pip install -e .
python -m pydlshogi2.train train.hcpe test.hcpe --gpu 0
```

`--gpu -1` runs on CPU. That is fine for a smoke test of the pipeline and
hopeless for real training — a 20×256 network on CPU is orders of magnitude too
slow.

## Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LoveKapibarasan/python-dlshogi2/blob/main/notebooks/train.ipynb)

```python
!pip install cshogi
!pip install git+https://github.com/LoveKapibarasan/python-dlshogi2.git
```

Colab's own constraints drive how you should configure the run:

- **Sessions are cut off.** Keep checkpoints on Drive and use `--save_interval`
  plus `--resume`; see [Training Pipeline](Training-Pipeline#preemption-safety).
- **The GPU varies** (T4 / L4 / A100). Batch size and `--amp` that fit one may
  not fit another — the GPU name is recorded in the metrics for exactly this
  reason, so a curve that looks anomalous can be checked against the hardware
  it ran on.
- **The local disk is ephemeral.** Write metrics next to the checkpoints on
  Drive, or they vanish with the session.

## Vast.ai (or any CUDA Linux box)

```bash
./vast_setup.sh                                       # once per instance
./vast_run.sh train    train.hcpe test.hcpe --gpu 0 --amp
./vast_run.sh selfplay checkpoints/checkpoint.pth sp.hcpe --games 1000 --gpu 0
./vast_run.sh rl       checkpoints/checkpoint.pth
```

`vast_setup.sh` creates `.venv`, installs the CUDA torch wheel from PyPI,
**swaps `onnxruntime` for `onnxruntime-gpu`** (falling back to the CPU build if
the GPU one is unavailable), installs the package editable, and prints a
`torch.cuda.is_available()` check. Run it from the repository root.

`vast_run.sh` bootstraps the venv on first use and launches the job under
`nohup`, so it survives an SSH disconnect:

```bash
tail -f logs/<mode>-*.log      # follow
kill $(cat logs/<mode>.pid)    # stop — a clean SIGTERM, so it checkpoints first
```

That `kill` is the same signal a preemption sends, and training handles it the
same way: finish the step, save, exit.

Anything after the mode is forwarded verbatim, so `--metrics`, `--amp`,
`--save_interval` and the rest all work. The RL loop is configured through its
environment variables instead (`ITERATIONS`, `GAMES`, `PLAYOUTS`, …).

### Spot instances

The whole preemption story matters most here. Before starting a long run:

- `--save_interval 1000` and a **single rolling** `--checkpoint` path.
- `--metrics` pointing next to the checkpoints, so one `scp` retrieves both.
- A fixed `--run_id` if you want the resumed run to appear as one run.
- `RL_RUN_ID` for the same reason on `rl_loop.sh`.

### Retrieving results

```bash
scp -r vast:python-dlshogi2/checkpoints .
scp -r vast:python-dlshogi2/rl/metrics  ./rl-metrics
DLSHOGI_METRICS_DIR=./rl-metrics streamlit run dashboard/app.py
```

The dashboard scans recursively, so several downloaded runs can be dropped into
one directory and compared side by side.

## Dashboard host

The dashboard has no GPU requirement and does not import torch (except on
demand, to read a checkpoint's architecture). Run it on a laptop against
downloaded metrics rather than on the GPU box.

---

See also: [Training Pipeline](Training-Pipeline),
[Metrics and Dashboard](Metrics-and-Dashboard), [Troubleshooting](Troubleshooting)
