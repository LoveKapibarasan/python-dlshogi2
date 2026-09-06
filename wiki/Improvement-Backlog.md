# Improvement Backlog

Ideas for making the engine stronger, ranked by what they are expected to be
worth against what they cost. This page is the **single source of truth** for
the list: a GitHub issue carries the discussion, the dashboard shows the
measurement, and both key on the `ID` column here.

How an entry moves through this page:

```
提案 (backlog に追記 + Issue 作成)
  └─ 実装 (branch)
       └─ 計測 (python -m pydlshogi2.match --experiment EXP-00N --issue N)
            ├─ SPRT accept → merge、状態を「採用」に、Experiment Log に結論を書く
            └─ SPRT reject → 状態を「棄却」に。棄却した理由こそ残す価値がある
```

The `--experiment` id passed to the match is what joins a row here to its
measured Elo in the dashboard's **改善案** tab, so it must match the `ID` cell
exactly. See [Evaluation and Rating](Evaluation-and-Rating) for how a match is
run and how to read its verdict.

## Backlog

| ID | 改善案 | 種別 | 期待 Elo | コスト | 状態 | Issue |
|----|--------|------|----------|--------|------|-------|
| EXP-001 | `select_max_ucb_child` の numpy 呼び出し削減と訪問済み方策和の差分更新 | search | +30〜60 → **実測 +123 ± 62** | 小 | **採用** | [#5](https://github.com/LoveKapibarasan/python-dlshogi2/issues/5) |
| EXP-002 | MCTS solver: 証明済みの勝ち/負けを親へ厳密に伝播する | search | +20〜60 | 中 | 未着手 | [#6](https://github.com/LoveKapibarasan/python-dlshogi2/issues/6) |
| EXP-003 | `rl_loop.sh` に昇格ゲートを入れる (前チェックポイントに勝てなければ昇格しない) | pipeline | 退行の防止 | 小 | 未着手 | [#7](https://github.com/LoveKapibarasan/python-dlshogi2/issues/7) |
| EXP-004 | 探索パラメータ (`c_puct` / `fpu_reduction` / `temperature`) の総当たり調整 | tuning | +0〜80 → **実測 c_puct は空振り** | 中 (GPU時間) | **棄却** | [#8](https://github.com/LoveKapibarasan/python-dlshogi2/issues/8) |
| EXP-005 | `batchsize` と virtual loss の見直し | tuning | +0〜30 | 小 | 未着手 | [#9](https://github.com/LoveKapibarasan/python-dlshogi2/issues/9) |
| EXP-006 | 局面評価の呼び出し経路を JIT 化する (numba / Cython) | search | +20〜60 (EXP-001 後の実測で下方修正) | 中 | 未着手 | [#10](https://github.com/LoveKapibarasan/python-dlshogi2/issues/10) |
| EXP-007 | 特徴量生成と `make_move_label` のベクトル化 | search | +5〜15 | 小 | 未着手 | [#11](https://github.com/LoveKapibarasan/python-dlshogi2/issues/11) |
| EXP-008 | 詰み探索の深さをルート以外にも広げる | search | +10〜40 | 中 | 未着手 | [#12](https://github.com/LoveKapibarasan/python-dlshogi2/issues/12) |
| EXP-009 | ネットワークの再学習 (20×256 SE, Floodgate 全体) | training | +200 以上 | 大 (GPU が足りない) | 保留 | [#13](https://github.com/LoveKapibarasan/python-dlshogi2/issues/13) |

## なぜこの順番なのか

順位は「期待 Elo ÷ 必要な GPU 時間」で付けている。手持ちの GPU が
[Environments](Environments) のとおり ollama と共有で数百 MB しか空いていない
以上、**モデルを鍛え直す案は最後**に回さざるを得ない。逆に、探索の改善は
モデルを固定したまま A/B できるので、同じ GPU 時間で何倍も多くの実験が回る。

### 探索は GPU ではなく Python で詰まっている

`checkpoints/checkpoint.pth` (10×192) で 1,500 playout を計測した内訳
(`eval_node` の前後で `time.process_time()` を読む):

| 内訳 | 割合 |
|------|------|
| 木の操作 (選択・展開・バックアップ) | **79 %** |
| ニューラルネット評価 (`eval_node`) | 21 % (うち順伝播 15 %) |

`nvidia-smi` の GPU 使用率はほぼ 0 % で、実測は数百 playout/s。
つまりこのエンジンは GPU ではなく **CPU 上の Python でボトルネックになっている**。
ONNX 化や TensorRT、より大きな `batchsize` は 21 % の部分にしか効かない。
EXP-001 と EXP-006 が上位にいるのはこのためで、探索が速くなれば同じ持ち時間で
より深く読める＝そのまま棋力になる。

木の操作の中では `select_max_ucb_child` が突出していた。`timeit` で直接計った
1 回あたりの時間は **約 145 マイクロ秒**で、しかも**合法手が 12 手でも 120 手でも
ほぼ変わらない**。計算量ではなく、40 要素程度の配列に対する numpy 呼び出しの
オーバーヘッドが支配的だったということ。EXP-001 はここを叩いた。

> **計測の落とし穴 (踏んだので記録)。** cProfile は自身のオーバーヘッドを
> 呼ばれた側に計上するので、34,000 回呼ばれる関数の秒数は大きく膨らむ。
> また作業台は ollama と CPU を共有していて、同じ探索が 6.5 秒から 11.9 秒まで
> 振れる。倍率を語るときは `timeit` か `time.process_time()` を使い、
> A/B は**交互に走らせて最小値で比べる**こと。詳細は [MCTS](MCTS) を参照。

### EXP-001 実施後の内訳

EXP-001 を入れると比率が変わり、次にどこを叩くべきかも変わる:

| 内訳 | 最適化前 | 最適化後 |
|------|---------|---------|
| 木の操作 | 79 % | 57〜62 % |
| ニューラルネット評価 | 21 % | 38〜43 % |

木の操作が支配的ではなくなるので、**EXP-006 (JIT 化) の上積みは当初の見積もりより
小さい**。逆にニューラルネット側 (EXP-005 の batchsize、推論バックエンド) の
相対的な価値は上がる。見積もりは実測が出るたびに直す。

### モデルの再学習 (EXP-009) を保留にしている理由

同梱の `checkpoints/checkpoint.pth` は Floodgate 2020 を 3 エポック学習した
10×192 (SE なし) で、明らかに伸びしろが大きい。README の既定である 20×256 SE を
本気で学習させれば他のどの案より大きな差が出る — ただしそれは、この作業台の
RTX 3050 (空き 300〜800 MB) では現実的でない。Colab か Vast.ai を使える時に
回すべき案として、消さずに保留にしてある。

## 追加するときの書き方

- **ID は連番** (`EXP-0NN`)。一度振った ID は再利用しない — 計測記録が指している。
- **期待 Elo を先に書く。** 外れたときに「何を見誤ったのか」が残る。当てるための
  欄ではなく、後から自分の見積もりを較正するための欄。
- **コストは GPU 時間で見積もる。** 実装が 30 分でも計測に 6 時間かかるなら「中」。
- **棄却も残す。** 効かなかった案は、次に同じ思いつきをした人への一番の情報。
  行は消さず、状態を「棄却」にして [Experiment Log](Experiment-Log) に結論を書く。
- **狙う効果量を先に決める。** この作業台の分解能は **100 局で ±60 Elo**。
  それより小さい差を狙う実験は、何局回しても結論が出ないので設計として成立しない
  (EXP-004 で実証済み)。詳細は [Evaluation and Rating](Evaluation-and-Rating)。

---

See also: [Evaluation and Rating](Evaluation-and-Rating),
[Experiment Log](Experiment-Log), [MCTS](MCTS),
[Metrics and Dashboard](Metrics-and-Dashboard)
