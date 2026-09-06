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

## 2026-09-05 — EXP-001: 探索の高速化は +123 Elo になった

- **experiment**: `EXP-001` ([#5](https://github.com/LoveKapibarasan/python-dlshogi2/issues/5))
- **run_id**: `20260905-122247-9d50ea29`
- **commit**: `4806a5e` (`perf/fast-puct`) 対 `cd2277c` (`feat/match-arena-and-rating`)
- **hardware**: RTX 3050 (作業台。ollama と共有、load average 3〜7)
- **model**: 両者とも同じ `checkpoints/checkpoint.pth` (10×192, SE なし)
- **question**: `select_max_ucb_child` の numpy 呼び出しを減らした高速化は、
  実際の棋力になるのか。

### 結果

**固定時間 1 手 1 秒、100 局:**

```
W-L-D      : 58-24-18
score      : 0.6700  (引き分け率 18.0%)
pairs      : 50 ペア [0, 1/2, 1, 1 1/2, 2] = [2, 1, 26, 3, 18]
Elo        : +123.0 +/- 62.4   95% CI [+64.5, +189.2]
LOS        : 99.6%
verdict    : engine1 is stronger (95% 区間が 0 を含まない)
```

ペアの内訳が効いている。**50 ペア中 18 ペアで 2 連勝、2 連敗はわずか 2 ペア**。
26 ペアは 1 勝 1 敗で、これは互角のときに必ず起きる形。

**固定プレイアウト (200 playout) では、完了した 3 ペアがすべて 1 勝 1 敗、
score ちょうど 0.5000。** これは強さの証拠ではなく、**挙動を一切変えていない**ことの
確認である。ここが 50 % から外れていたら、それは高速化ではなく別の探索になっている。

**探索速度 (CPU 時間、1500 playout、BASE と FAST を交互に走らせた最小値):**

| | 最適化前 | 最適化後 | 比 |
|---|---------|---------|---|
| `select_max_ucb_child` (合法手 40) | 155.1 us | 19.1 us | 8.1x |
| 探索全体 | 6.47 s | 3.29 s | **1.96x** |

交互 6 回の比は 1.51x 〜 2.45x で、**全回が改善側**。

### 結論

**採用 (merge)。** 1 手 1 秒という短い持ち時間では 1 手あたりの読みの量が
数百 playout しかないので、そこが 2 倍になれば +123 Elo は妥当な大きさ。
持ち時間を長くすれば効果は逓減するはずで、この数字を「どんな条件でも +123」と
読んではいけない。

副産物として、木の操作が探索時間の 79 % から約 60 % に下がり、
ニューラルネット評価が 21 % から約 40 % になった。**次に叩くべき場所が変わった**ので、
[Improvement Backlog](Improvement-Backlog) の EXP-006 (JIT 化) の期待値を
+60〜120 から +20〜60 に下方修正した。

### 測定で踏んだ罠

この実験は**結論を 2 回間違えた**。どちらも計測方法の問題だったので記録しておく。

1. **cProfile の絶対値を信用した。** cProfile は 1 回の呼び出しごとに自身の
   オーバーヘッドを呼ばれた側の tottime に計上する。`select_max_ucb_child` は
   1 回の探索で 34,000 回以上呼ばれるので、秒数が大きく膨らんでいた。
   方向は正しかったが、倍率は `timeit` で計り直すまで信用できなかった。
2. **共有マシンで壁時計を使った。** 作業台の CPU は ollama と共有で、
   同じ探索が 6.5 秒から 11.9 秒まで振れる。最初に取った A/B は
   「最適化後のほうが 23 % **遅い**」という結論を出した。
   `time.process_time()` に切り替え、BASE と FAST を交互に走らせて
   最小値で比べて初めて、一貫した比が出た。

なお対局そのものは、固定時間なら両者が同じ持ち時間を交互に使うので、
マシンの負荷が乗っても比較としては公平である。実際 174 分かかったが結論は明快だった。

- **git_dirty**: `true` と記録されているが、これは `csa/` と `openings.txt` が
  未追跡だったためで、**追跡ファイルの変更はない**。commit から再現できる。
- **棋譜**: 100 局すべて CSA で保存 (`csa/`)。

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
