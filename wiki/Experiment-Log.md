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

## 2026-09-25 — EXP-002: MCTS solver は勝敗を一度も変えなかった

- **experiment**: `EXP-002` ([#6](https://github.com/LoveKapibarasan/python-dlshogi2/issues/6))
- **run_id**: `20260925-215633-2e57a180`
- **commit**: `967b193` (`feat/mcts-solver`) 対 `54c7b06` (main)
- **hardware**: RTX 2060 6GB (新しい作業台。**専有**、ollama なし、5 コア)
- **model**: 両者とも `checkpoints/checkpoint.pth` (10×192, SE なし)
- **条件**: **固定 400 プレイアウト**、64 定跡を先後入れ替え、ペア計測、SPRT H0=0 / H1=+60
- **question**: 証明済みの勝ち負けを親へ厳密に伝播すれば、終盤の読み負けが減るのでは。

### 実装

- 子が 1 つでも「相手の負け」と証明されたら親は勝ち、全ての子が作成済みで
  「相手の勝ち」なら親は負け (`propagate_proof`)。引き分けは伝播しない
- 「相手の勝ち」と証明された子は平均値を厳密な負け (0) に揃える
- ルートでは証明済みの勝ちを必ず指し、証明済みの負けは訪問数によらず指さない。
  ルートが証明されたらその時点で探索を打ち切る

単体では狙いどおりに動く。ルート詰み探索を 1 手に絞った 3 手詰めの局面で、
main は 300 プレイアウトで詰みを逃して 2e2f を指すが、solver は 210 プレイアウトで
ルートを証明して 9e8e で詰ます (`tests/test_mcts_solver.py`)。

### 結果

```
games      : 30 (SPRT で打ち切り)
W-L-D      : 12-12-6
pairs      : 15 ペア [0, 1/2, 1, 1 1/2, 2] = [0, 0, 15, 0, 0]
Elo        : -0.0 +/- 47.3   95% CI [-47.3, +47.3]
SPRT       : llr=-3.070 H0=0 H1=60 -> reject
time       : 16 分、32 秒/局
```

**15 ペアすべてが 1 勝 1 敗。** ただし「何も変えていない」わけではない。
ペアの 2 局を指し手で比べると:

```
指し手まで完全一致 : 7 ペア
途中で分岐         : 8 ペア — 分岐はすべて 77〜144 手目、勝敗が変わったペアは 0
```

### 結論: **棄却。merge しない。**

solver が手を変えるのは**勝敗がすでに決まった局面だけ**だった。考えれば当然で:

- ルートでは既に 7 手詰め探索 (`mate_root_ply`) が走っていて、短い詰みは solver の
  出番の前に見つかる
- 木の中の証明は展開時の 3 手詰めが起点なので、証明が起きるのは
  価値ネットワークもすでに 0.9 以上/0.1 以下を返している局面に偏る
- 400 プレイアウトでは木が浅く、ルートまで届く「深い証明」がほとんど作られない

**勝ちを早く・確実に詰ます効果はあるが、勝敗は変えない。** 期待値 +20〜60 は
過大だった。実装は `feat/mcts-solver` に残してある — EXP-008 (木の中の詰み探索を
深くする) と組み合わせれば証明が起きる局面が前倒しになるので、そのときに
土台として再評価する価値はある。

### 踏んだ罠: 例外で bestmove が返らず対局が止まった

最初の計測は 14 局目で**無言で止まった**。ルートで訪問済みの手がすべて
負けと証明され、未訪問の手しか残っていないとき、着手選択が訪問数 0 の手を選び、
勝率 `0/0 = nan` → `int(nan)` で `ValueError`。これが `go` を走らせている
スレッドの中で起きるので、**例外は `Future` に握りつぶされ、bestmove が出ないまま
両エンジンが待ち続ける**。修正は `967b193`。

教訓は 2 つ:

- 探索の変更は、対局 harness に載せる前に**同一プロセスで数十局指させて例外を
  拾う**。この修正後、40 局 4,137 手を例外なしで確認してから再計測した
- `BasePlayer.run` の `go` スレッドの例外は今も握りつぶされる。
  止まったら**エンジンの CPU が 0 % かどうか**を見れば、探索中か死んでいるかが分かる

- **時間**: この作業台は 1 局 32 秒で、EXP-004 (RTX 3050、共有、154 秒/局) の
  約 **5 倍速い**。同じ一晩で 5 倍の局数を回せるので、「±60 Elo の壁」は
  ここでは局数を増やして下げられる。
- **git_dirty**: `openings.txt` と `csa/` が未追跡なだけで、追跡ファイルの変更はない。
- **metrics**: `match-EXP-002-20260925-215632.jsonl`。ハングした初回
  (`match-EXP-002-20260925-210303`, 13 局) はバグ入りの版なので計測から外した。

---

## 2026-09-06 — EXP-004: c_puct の調整は空振り。既定値 1.0 で妥当だった

- **experiment**: `EXP-004` ([#8](https://github.com/LoveKapibarasan/python-dlshogi2/issues/8))
- **commit**: `76ee651` (main、EXP-001 適用後)
- **model**: 両者とも `checkpoints/checkpoint.pth` (10×192, SE なし)
- **条件**: **固定 400 プレイアウト**、64 定跡を先後入れ替え、ペア計測
- **question**: 方策ネットが弱いのだから、探索係数を振れば拾えるものがあるのでは。

### 結果

| c_puct | 局数 | W-L-D | score | Elo | 95% CI | 判定 |
|--------|------|-------|-------|-----|--------|------|
| 0.7 vs 1.0 | 100 | 41-44-15 | 0.485 | -10.4 ± 60.0 | [-70.8, +49.3] | 互角 |
| 2.5 vs 1.0 | 27 | 2-20-5 | 0.173 | -271.7 | [-665.6, -127.0] | **大幅に悪い** |

ペアの内訳が分かりやすい:

```
c_puct=0.7 : [8, 6, 26, 1, 9]   ほぼ左右対称 = 差が無い
c_puct=2.5 : [8, 1, 4, 0, 0]    13ペア中8ペアが既定値側の2連勝、逆は0
```

### 結論: **棄却。既定値 1.0 を変えない。**

- 下げても (0.7) 変わらない
- 上げると (2.5) **壊滅的に悪くなる** — 400 プレイアウトしかない予算で 50 手前後に
  訪問を散らせば、どの手も読み切れなくなるので当然ではある
- したがって **c_puct=1.0 は既に良い位置にある**。この方向に伸びしろは無い

**1.5 は測っていない。** 崖が 1.0 と 2.5 の間のどこにあるかは分かるが、
1.0 が既に崖の安全側にいる以上、**崖の位置を特定してもエンジンは強くならない**。
GPU 時間の使い道として割に合わないので飛ばした。

### この実験の本当の収穫は検出限界のほう

**当初の設計は成立していなかった。** Issue #8 には「SPRT (H0=0, H1=+15) で早期に
打ち切る」と書いたが、これは動かない。差が本当にゼロのとき、LLR は 1 局あたり
**約 -0.001** しか動かない:

```
30局時点で llr = -0.04  →  下限 -2.94 に到達するのに約2000局
```

**100 局の 95% 区間は ±60 Elo。** つまりこの作業台で見つけられるのは
**±60 Elo より大きい差だけ**であり、「+15 Elo の改善を探す」実験は設計として
破綻している。実際、c_puct=2.5 の -272 Elo は検出限界の 4.5 倍あったので
**27 局で決着した**。一方 c_puct=0.7 は 100 局かけても何も言えなかった。

今後この作業台で実験を設計するときの原則:

- **狙う効果量を先に決める。** ±60 Elo 以下を狙う実験はここでは回さない
- 大きい効果が期待できる案 (探索アルゴリズムの変更、モデルの入れ替え) を優先する
- 小さい差を詰めたいなら、局数ではなく **持ち時間を短くして局数を稼ぐ** か、
  そもそも別のマシンを用意する

- **時間**: c_puct=0.7 が 257 分 (154 秒/局)。作業台が KomoringHeights のビルドと
  ollama で load 12 まで上がっていた。固定プレイアウトなので**結果は負荷の影響を
  受けない**が、所要時間は 1.5 倍になった。
- **metrics**: `match-EXP-004-cpuct70-*.jsonl`, `match-EXP-004-cpuct250.jsonl`

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
