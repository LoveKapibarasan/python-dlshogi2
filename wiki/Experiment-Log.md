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

## 2026-09-26 — EXP-006: 探索を C++ に移して 6.6 倍速、main に 10-0

- **experiment**: `EXP-006` ([#10](https://github.com/LoveKapibarasan/python-dlshogi2/issues/10))
- **commit**: `87eacec` / `c35f84f` (`feat/native-search`) 対 `54c7b06` (main)
- **hardware**: RTX 2060 6GB、Xeon E5-2630 v4 (対局中は自己対局を停止)
- **model**: 両者とも `checkpoints/checkpoint.pth` (10×192)

### 何をしたか

`MCTSPlayer.search` 以下 (選択・展開・詰み/千日手/入玉判定・評価の反映・バックアップ)
を **C++ に移植**した (`pydlshogi2/uct/native/uct.cpp`、ctypes で読む)。
盤面は cshogi 自身の C++ ソースを使うので、指し手の数値も合法手の順番も同一。
ニューラルネットだけ Python に残し、C++ からバッチごとに呼び戻す。

1. **移植は演算単位で同じ。** float32/float64 の丸めまで Python 版と同じ順に書いた。
   fp32 推論なら、ルートの訪問数が Python 版と **完全一致** する
   (`tests/test_native_search.py`)。移植中に入力特徴量の持ち駒プレーンの
   並びを 1 か所間違えたが (歩の最大枚数 8 と 18)、この比較が 32 プレイアウトで捕まえた
2. **推論は fp16 + channels-last + CUDA Graph。** バッチ 32 の 10×192 では、
   PyTorch の逐次実行はカーネル起動待ちが大半だった
3. **GPU と CPU を重ねる。** 2 つのバッチ枠を交互に使い、GPU が片方を評価している間に
   もう片方の降下を進める (dlshogi と同じ)

| 実装 | playout/s (4 局面 × 3,000) | 選んだ手 |
|------|------------------------------|----------|
| main (Python 探索) | 1,442 | 基準 |
| C++ 探索 + fp32 (Python と同一の木) | 2,950 | 同じ |
| + fp16 CUDA Graph | 6,164 | 同じ |
| + GPU/CPU の重ね合わせ | **9,546** | 同じ |

### 結果

main との対局 (1 手 1 秒、`87eacec` 時点 = 6,164 playout/s 版):

```
games      : 10 (SPRT で打ち切り)
W-L-D      : 10-0-0
pairs      : 5 ペア すべて 2 連勝
Elo        : 95% CI [+170, +3600]
SPRT       : llr=+3.097 H0=0 H1=100 -> accept
```

水匠10 (1 スレッド、10 万ノード/手) との対局 (`c35f84f`、9,546 playout/s 版):

| | W-L-D | score | Elo |
|--|-------|-------|-----|
| main (GAP 計測) | 3-16-1 | 0.175 | 約 -270 |
| **native** | **13-7-0** | 0.650 | 約 +110 (95% CI [-39, +310]) |

同じ相手に対して **約 +380 Elo** (各 20 局なので粗い)。
native 側の対局の序盤数局は、テストの CPU 負荷と重なっていた。

### 結論: **採用。** 渡されたコードを大きく上回った

- 読みの量が 6.6 倍になり、それがそのまま棋力になった。モデルは同じ
- 次のボトルネックは GPU (推論)。10×192 をバッチ 32 で回すと RTX 2060 は
  ほぼ埋まっている
- 自己対局 (`selfplay.py`) はまだ Python 探索なので、ここにも載せれば
  ゼロからの強化学習の生成速度も数倍になる

- **metrics**: `match-EXP-006-20260926-140949.jsonl`, `match-GAP-native-n100000-*.jsonl`

---

## 2026-09-26 — GAP: YaneuraOu + 水匠10 との差はおよそ 700〜900 Elo

- **目標**: 同じ PC 上の YaneuraOu + 水匠に勝ち越すこと。まず差を測った。
- **相手**: YaneuraOu 9.80git (NNUE halfkp_512x2_8_64、AVX2、g++ でビルド) + 水匠10 (`suisho10_250413`)。
  定跡オフ (`BookFile=no_book`)、`NetworkDelay=0`
- **こちら**: main (`54c7b06`)、`checkpoints/checkpoint.pth` (10×192)
- **条件**: 1 手 1 秒、64 定跡を先後入れ替え、各 20 局
- **hardware**: Xeon E5-2630 v4 (5 コア) + RTX 2060

| 水匠の条件 | W-L-D (こちら視点) | score | Elo (概算) |
|------------|--------------------|-------|------------|
| 本気: 4 スレッド、1 手 1 秒 (約 100 万ノード/手) | **0-20-0** | 0.000 | 測定不能 (CI 上限 -290) |
| 1 スレッド、**10 万**ノード/手 | 3-16-1 | 0.175 | 約 -270 |
| 1 スレッド、**1 万**ノード/手 | 18-1-1 | 0.925 | 約 +440 |

### 読み方

こちらの棋力は「1 手 1〜10 万ノード読む水匠」の間にある。本気の水匠は 1 手に
約 100 万ノード (4 スレッドで約 92 万 nps) 読むので、10 万ノードからさらに
約 10 倍。αβ 探索の倍々の伸び (1 倍化あたり +100〜150 Elo 程度) で外挿すると、
**本気の水匠との差は 700〜900 Elo** と見積もる。20 局ずつなので粗い数字である。

### 結論: 探索の小改善では届かない。**判断の質 (モデル) を替える**

- バックログの探索改善 (EXP-002/008: 実測 ±0 / +17) を全部足しても 100 Elo に届かない
- こちらは Python の探索で 1 手あたり 1,500 プレイアウト程度。**1 局面あたりの判断の質**
  (ネットワーク) が弱いので、同じ読みの量ならまず負ける
- そこで **水匠10 を教師にした蒸留** (EXP-010) に切り替える: 水匠同士の自己対局
  (2 万ノード/手、序盤 0〜8 手ランダム) の各局面の最善手・評価値・勝敗で学習し直す。
  生成は `utils/gen_teacher.py`、4 プロセスで約 45 局面/秒 (約 390 万局面/日)

### 踏んだ罠: `match.sh` の `FOREGROUND=1` が対局を 2 回走らせていた

`exec cmd | tee` はパイプラインの中なので exec されず、対局が終わるとスクリプトが
そのまま下の `nohup` に進み、**同じ対局をもう一度バックグラウンドで起動**していた。
しかも同じログ名・同じ metrics ファイル。連続実行のラダーで 2 段が並走して発覚した。
修正は `fix/match-foreground`。並走した段の記録は捨てて測り直した。

- **metrics**: `match-GAP-suisho10-20260926-025233.jsonl` (本気)、
  `match-GAP-suisho10-n100000-*`, `match-GAP-suisho10-n10000-*`

---

## 2026-09-26 — EXP-008: 木の中の 5 手詰めは +17 ± 59 で決着せず

- **experiment**: `EXP-008` ([#12](https://github.com/LoveKapibarasan/python-dlshogi2/issues/12))
- **run_id**: `20260926-014114-d2cec77f`
- **commit**: `8393d5d` (`feat/mate-tree-ply`、`mate_tree_ply=5`) 対 `54c7b06` (main、3 手)
- **hardware**: RTX 2060 6GB (専有)、Xeon E5-2630 v4 5 コア
- **条件**: **固定時間 1 手 0.5 秒**、64 定跡を先後入れ替え、ペア計測、SPRT H0=0 / H1=+60
- **question**: ノード展開時の詰み探索を 3 手から 5 手に深くすると、探索速度の損を上回るか。

### 事前の計測: 詰み探索のコスト

実戦の局面 (EXP-002 の棋譜) で `mate_move(N)` 単体:

| 深さ | 1 回あたり (100 手目以降) | 詰みを見つけた割合 |
|------|---------------------------|--------------------|
| 3 | 16 us | 1/300 |
| 5 | 43 us | 3/300 |
| 7 | 227 us (最大 5 ms) | 8/300 |

探索全体では (終盤 12 局面 × 800 playout、CPU 時間、交互 3 回の最良値):
3 手 1,509 → 5 手 1,478 (**-2 %**) → 7 手 1,269 (**-16 %**) playout/s。
5 手はほぼ無料、7 手は無条件に入れると速度を明確に削る。まず 5 手を測った。

### 結果

```
games      : 100
W-L-D      : 50-45-5
pairs      : 50 ペア [0, 1/2, 1, 1 1/2, 2] = [8, 0, 31, 1, 10]
Elo        : +17.4 +/- 59.2   95% CI [-41.3, +77.1]
SPRT       : llr=-0.822 -> continue
time       : 71 分、43 秒/局
```

### 結論: **保留。結論は出ていないし、これ以上局数を足す価値もない。**

- 期待値 +10〜40 に対して実測 +17。**分解能 (±60) の中**なので何も言えない
- 固定時間では秒単位の揺らぎで手が変わるので、ペアの 2 局が一致したのは
  50 ペア中 5 ペアだけ (分岐の中央値 53 手目)。EXP-002 の固定プレイアウトと違い、
  「どこで変更が効いたか」を棋譜から追うことはできない
- 目標が YaneuraOu + 水匠との対戦になった今 (下記 GAP 計測)、
  ±20 Elo を詰めるために数百局を使う理由がない。`mate_tree_ply` はオプションとして
  残す価値がある (既定は 3 のまま、挙動不変)

- **metrics**: `match-EXP-008-20260926-014114.jsonl`

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
