# Evaluation and Rating

How a change is judged. Source:
[`pydlshogi2/match.py`](https://github.com/LoveKapibarasan/python-dlshogi2/blob/main/pydlshogi2/match.py),
[`pydlshogi2/rating.py`](https://github.com/LoveKapibarasan/python-dlshogi2/blob/main/pydlshogi2/rating.py),
[`utils/make_opening_book.py`](https://github.com/LoveKapibarasan/python-dlshogi2/blob/main/utils/make_opening_book.py).

Training curves say a model fits its data. They do not say it plays better, and
a change to the *search* moves no loss at all. The only instrument that answers
"is this stronger?" is a match, so every proposal in the
[Improvement Backlog](Improvement-Backlog) ends up here.

## The short version

```bash
# 1. 定跡を作る (一度だけ。決定的な探索を分岐させるために必須)
python utils/make_opening_book.py checkpoints/checkpoint.pth openings.txt \
    --lines 64 --plies 12

# 2. main のワークツリーを用意して、ブランチと対局させる
git worktree add ../wt-main main

python -m pydlshogi2.match \
    --engine1 ./usi_engine.sh          --name1 fast-puct \
    --engine2 ../wt-main/usi_engine.sh --name2 main \
    --options1 modelfile=checkpoints/checkpoint.pth \
    --options2 modelfile=checkpoints/checkpoint.pth \
    --games 100 --byoyomi 1000 --opening openings.txt \
    --sprt --elo1 20 \
    --metrics metrics/match-EXP-001.jsonl --experiment EXP-001 --issue 7
```

The verdict is printed at the end and recorded in the metrics file, where the
dashboard's **レーティング** and **改善案** tabs pick it up.

## なぜ定跡ファイルが要るのか

この実装の MCTS は**決定的**である。同じ局面・同じプレイアウト数なら必ず同じ手を
返す。したがって平手の初期局面から始めた 2 つのエンジンは毎回まったく同じ将棋を
指し、「100 局」は 1 局を 100 回数えただけになる。

`utils/make_opening_book.py` は方策ヘッドだけ (探索なし、1 手 1 回の順伝播) から
手を温度付きサンプリングして開始局面を散らす。完全にランダムな手で散らすことも
できるが、それでは「無茶苦茶な局面をどう処理するか」を測ることになってしまう。

## 先後を入れ替えたペアで数える

`cshogi.cli` は 1 局ごとに先後を入れ替え、同じ定跡を 2 局続けて使う。この
**ペアが計測の単位**であり、それには効く理由がある。

このリポジトリの対局harnessを、同じエンジン同士で走らせた実際の出力:

```
self-a vs self-b start.
まで71手で先手の勝ち          ← 定跡A、engine1 が先手
self-b vs self-a start.
まで71手で先手の勝ち          ← 定跡A、engine2 が先手。まったく同じ71手
```

同じ強さのエンジンは、どのペアでも必ず 1 勝 1 敗になる。1 局ずつ独立に数える
式 (三項分布) はここで「勝率 50 %、誤差 ±360 Elo」と報告してしまうが、実際には
**ばらつきはゼロ**で、この結果は「互角」を強く示している。

そこで `PairedMatchStats` は 1 局ではなく 1 ペアを 1 観測とし、0 / ½ / 1 / 1½ / 2
点の五項分布として分散を見積もる。これはチェスのテスト基盤が pentanomial と呼ぶ
方式で、**同じ結論に必要な対局数がおよそ半分になる**。GPU が他の用途と共有で
1 局に 1 分かかる状況では、この差はそのまま実験が回るかどうかの差になる。

1 局ずつ数えたい場合 (定跡を使わない、あるいはエンジンが非決定的な場合) は
`--unpaired` を付ける。

## 100 局で何が分かって、何が分からないか

これが一番大事な注意点である。**100 局は思っているほど分からない。**

| 実際の差 | 100 局(ペア計測)で SPRT が決着する確率 | 備考 |
|----------|------------------------------------------|------|
| +100 Elo | ほぼ確実 | 30〜40 局で accept に届く |
| +50 Elo | 高い | |
| +20 Elo | 五分 | 200〜400 局欲しい |
| +5 Elo | ほぼ無理 | 1,000 局規模の話 |

1 局ずつ数える場合、100 局の 95 % 信頼区間はおよそ **±70 Elo** ある。つまり
「100 局で 55 勝 45 敗だったので改善」は**統計的に何も言っていない**。

### 実測: この作業台の分解能は 100 局で ±60 Elo

机上の話ではなく、実際に測った 2 例。どちらも同じ harness、同じ 100 局の予算。

| 実験 | 差 | 局数 | 結果 |
|------|----|------|------|
| EXP-001 (探索の高速化) | 大きい | 100 | **+123.0 ± 62.4**、LOS 99.6% → 決着 |
| EXP-004 (c_puct 0.7) | 無い | 100 | -10.4 ± 60.0 → **何も言えない** |
| EXP-004 (c_puct 2.5) | 非常に大きい | **27** | -271.7、95% CI [-666, -127] → **27 局で決着** |

読み取れることは 2 つ:

1. **±60 Elo が分解能の壁。** これより小さい差は 100 局では出ない。
2. **効果が大きければ局数は要らない。** c_puct=2.5 は分解能の 4.5 倍悪かったので
   27 局で結論が出た。逆に言えば、**100 局回してもまだ決着しないなら、
   その差は実用上どうでもいい大きさ**である可能性が高い。

さらに、SPRT の下側 (「改善なし」と判定する側) は小さい仮説では機能しない。
H0=0 / H1=+15 で差が本当にゼロのとき、LLR は 1 局あたり約 -0.001 しか動かず、
下限 -2.94 に届くのに **2000 局以上**かかる。**H1 は分解能より大きく取ること。**

対処は 3 つある。

1. **SPRT で早期に打ち切る** (`--sprt`)。決着がついた時点で止まるので、明らかな
   退行は 30 局程度で分かり、浮いた時間を次の実験に回せる。
2. **ペアで数える** (既定)。上記のとおり必要局数がおよそ半分になる。
3. **持ち時間を短くして局数を増やす。** `--byoyomi 500` は `--byoyomi 1000` の
   倍の局数を同じ時間で稼げる。短い持ち時間で出た差が長い持ち時間でも残るとは
   限らないが、まず差があるかを知るには十分なことが多い。

## 固定プレイアウトと固定時間の使い分け

`--playouts` と `--byoyomi` は**別のことを測っている**。取り違えると結論が逆に
なるので注意する。

| 条件 | 何を測っているか | 使うべき場面 |
|------|------------------|--------------|
| `--playouts N` (固定探索量) | 1 プレイアウトあたりの質。速度は結果に影響しない | モデル同士の比較、RL ループの昇格判定 |
| `--byoyomi T` (固定時間) | 実戦の棋力。速度も質もまとめて効く | 探索の高速化、最終判断 |

**探索を速くしただけの変更は、固定プレイアウトでは必ず 50 % になる。** それは
失敗ではなく、挙動を変えていないことの証明である。高速化の効果は固定時間でしか
現れない。逆に、モデルを比べたいときに固定時間を使うと、たまたま速い側が有利に
なってしまう。

したがって高速化の変更は **2 本立て**で検証するのが正しい:

```bash
# 1) 固定プレイアウト: 50% であるべき (挙動が変わっていないことの確認)
python -m pydlshogi2.match ... --playouts 400 --games 40

# 2) 固定時間: 50% を超えるべき (速度が棋力になったことの確認)
python -m pydlshogi2.match ... --byoyomi 1000 --games 100 --sprt
```

## 結果の読み方

```
=== match result: fast-puct vs main ===
games      : 64 (stopped early by SPRT)
W-L-D      : 24-14-26
score      : 0.5781  (draw ratio 40.6%)
pairs      : 32 scored as [0, 1/2, 1, 1 1/2, 2] = [1, 5, 12, 11, 3]
Elo        : +54.7 +/- 41.2   95% CI [+14.1, +96.6]
LOS        : 96.4%
SPRT       : llr=+2.981 bounds=[-2.944, 2.944] H0=0 H1=20 -> accept
verdict    : engine1 is stronger; adopt the change
```

- **score** — 引き分けを 0.5 とした勝率。
- **pairs** — 五項分布の内訳。左端 (0 点) は「ペアで 2 敗」、右端 (2 点) は
  「ペアで 2 勝」。中央 (1 点) が多いほど互角に近い。
- **Elo ± 誤差** — 95 % 信頼区間の半幅。**区間が 0 をまたいでいたら、勝ち越して
  いても「分からない」が正しい結論。**
- **LOS** — engine1 のほうが強い確率。決着局のみから計算する。
- **SPRT** — `H0`(改善なし) と `H1`(採用に値する改善) の逐次検定。`accept` なら
  採用、`reject` なら棄却、`continue` なら局数が足りない。

`reject` のときだけ終了コードが 1 になるので、CI やスクリプトからそのまま
昇格の判定に使える。

## レーティング表

個々の対局は 2 つのエンジンの**差**しか教えてくれない。チェックポイントが
増えてくると「3 世代前と比べてどうなのか」を知りたくなるが、そのために過去の
組み合わせをすべて指し直すのは現実的でない。

`bradley_terry_ratings` は記録されたすべての対局を同時に説明する最尤レーティングを
求める (Bradley-Terry モデル、MM 法)。a>b と b>c の対局しかなくても a と c の差が
推定できるので、**基準となる 1 つのチェックポイントに対する相対値**として全体を
1 本の尺度に載せられる。ダッシュボードの **レーティング** タブがこの表を表示する。

一度も負けていないエンジンはレーティングが発散するので、平均的な相手との
仮想的な引き分け (`prior`) を少量入れて正則化してある。対局数が増えれば影響は
消える。

## 記録される内容

`--metrics` を付けると 1 局ごとに 1 レコード、最後に集計レコードが JSONL に
追記される。`--experiment` と `--issue` は run レコードに載り、これが
[Improvement Backlog](Improvement-Backlog) の行とダッシュボードの計測結果を
結びつける鍵になる。スキーマは [Metrics and Dashboard](Metrics-and-Dashboard) を参照。

途中で落ちた対局も読める。60 局で止まった 100 局マッチは、集計レコードこそ無い
ものの最新の 1 局レコードから状態を復元できるので、ダッシュボードには
`running` として 60 局分の結果が出る。

---

See also: [Improvement Backlog](Improvement-Backlog),
[Experiment Log](Experiment-Log), [MCTS](MCTS),
[Metrics and Dashboard](Metrics-and-Dashboard), [USI Engine](USI-Engine)
