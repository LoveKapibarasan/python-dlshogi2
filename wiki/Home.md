# python-dlshogi2 Wiki

Deep-learning shogi engine: a policy-value ResNet trained on game records,
improved by self-play, and played through Monte Carlo Tree Search over the USI
protocol.

## Where things are documented

This project keeps three kinds of documentation, and they deliberately do not
overlap. When adding something, pick by **who is reading it and why**:

| Place | Answers | Written in |
|-------|---------|------------|
| [README](https://github.com/LoveKapibarasan/python-dlshogi2/blob/main/README.md) | "How do I install this and run it?" — the shortest path from clone to a playing engine | `README.md`, kept short |
| [API reference](https://github.com/LoveKapibarasan/python-dlshogi2/tree/main/docs) (Sphinx) | "What does this function take and return?" — generated from in-source docstrings | docstrings + `docs/*.rst` |
| **This wiki** | "Why is it built this way, and how do I actually operate it?" — design background, environment-specific procedures, experiment records | `wiki/*.md` in the repo, synced to the wiki |

Rule of thumb: if it would go stale the moment the code changes, it belongs in a
docstring. If it explains a decision, a trade-off or a procedure, it belongs
here.

## Pages

**Design**

- [Architecture](Architecture) — input features, move labels, the ResNet and its heads
- [MCTS](MCTS) — PUCT, virtual loss, FPU, mate search, tree reuse, time management

**Operating the pipeline**

- [Training Pipeline](Training-Pipeline) — CSA → HCPE, the value target, preemption-safe training
- [Reinforcement Learning](Reinforcement-Learning) — what one `rl_loop.sh` iteration actually does
- [Metrics and Dashboard](Metrics-and-Dashboard) — the JSONL run history and the Streamlit dashboard
- [Environments](Environments) — Colab, Vast.ai and local GPU setup
- [USI Engine](USI-Engine) — registering the engine in a shogi GUI, PyTorch vs ONNX

**Records**

- [Experiment Log](Experiment-Log) — what was trained, with what settings, and what came out
- [Troubleshooting](Troubleshooting) — known pitfalls and their fixes

## Editing this wiki

The pages live in the main repository under `wiki/` and are pushed to the GitHub
wiki by `wiki/publish.sh`. Edit them there, in a normal pull request, so wiki
changes are reviewed alongside the code they describe. See
[wiki/README.md](https://github.com/LoveKapibarasan/python-dlshogi2/blob/main/wiki/README.md).

A GitHub wiki is itself a git repository, so it can also be cloned directly:

```bash
git clone git@github.com:LoveKapibarasan/python-dlshogi2.wiki.git
```

Edits made that way (or in the browser) will be **overwritten** by the next
`publish.sh` run, so treat `wiki/` in the main repo as the source of truth.
