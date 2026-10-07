# Human-like shogi AI (separate sub-project)

This directory is a **separate effort** from the strong self-play engine. The
goal here is *human-likeness*, not maximal strength: imitate how humans of a
given skill level play, in the spirit of [Maia Chess](https://maiachess.com/).

It does **not** touch the reinforcement-learning pipeline. It only reuses the
shared network (`pydlshogi2.network`) and feature encoding (`pydlshogi2.features`)
at training time; the data tooling here depends only on `cshogi`.

## 0. Shogi Wars KIF -> CSA (if your data is Shogi Wars .kif)

Shogi Wars exports are UTF-8 KIF with the player rank on `先手段級：` / `後手段級：`
lines and a bare `投了` ending that `cshogi` cannot interpret. `kif_to_csa.py`
handles all of this and unifies the corpus to CSA, mapping each dan/kyu rank to
an **ordinal** (higher = stronger) emitted as a floodgate-style rate line:

```
30級=1, 29級=2, ..., 1級=30, 初段=31, 二段=32, ..., 九段=39
```

```bash
python human/kif_to_csa.py ~/kifs/kif_data converted   # -> converted/shogiwars-*.csa
```

## 1. Build rank-bucketed data

The bands are given in **ordinals** (see the mapping above). For example
`--bands 31,33` makes three buckets: kyu (`<31`), 初段-二段 (`31-32`), 三段+ (`33+`).

```bash
# from the converted Shogi Wars CSA
python human/csa_to_hcpe_by_rating.py converted out --bands 31,33 --filter_moves 20

# or, for numeric-rated CSA (e.g. floodgate), use rating boundaries directly
python human/csa_to_hcpe_by_rating.py ~/csa out \
    --bands 1500,1800,2100,2400 --filter_moves 20 --test_ratio 0.05
```

Each position is bucketed by the **rating of the player to move**, and the
training target is the move that player actually chose. Output:

```
out/
  0000-1499/ {train,test}.hcpe
  1500-1799/ {train,test}.hcpe
  1800-2099/ {train,test}.hcpe
  2100-2399/ {train,test}.hcpe
  2400-up/   {train,test}.hcpe
```

Rating extraction requires the CSA records to carry `'black_rate:` / `'white_rate:`
comment lines (floodgate-style). Records without ratings are skipped unless you
pass `--allow_unrated` (they then go to an `unrated/` band).

### Nine single-rank bands, 3級 .. 六段

To train one model per rank from 3級 to 六段, give every ordinal from 28 (3級)
to 37 as an edge. That makes one band per rank, `0028-0028` (3級) through
`0036-0036` (六段), plus the `0000-0027` / `0037-up` catch-alls, which are not
trained:

```bash
python human/kif_to_csa.py ~/data/kif ~/data/csa --shards 256   # many shards: each is parsed in RAM
python human/csa_to_hcpe_by_rating.py ~/data/csa ~/human_data \
    --bands 28,29,30,31,32,33,34,35,36,37 --filter_moves 20
nohup ./human/train_all_bands.sh > ~/human_data/train_all.log 2>&1 &
```

| band dir | 0028 | 0029 | 0030 | 0031 | 0032 | 0033 | 0034 | 0035 | 0036 |
|---|---|---|---|---|---|---|---|---|---|
| rank | 3級 | 2級 | 1級 | 初段 | 二段 | 三段 | 四段 | 五段 | 六段 |

### Run the rank pipeline on Vast.ai

The Vast runner uses the same rank-filtered HCPE dataset and training pipeline.
For an unattended cloud run, first upload the bundle created below to the
configured R2 bucket, then set `VAST_API_TOKEN`, `R2_ACCESS_KEY`,
`R2_SECRET_ACCESS_KEY`, and `R2_ENDPOINT` in the local shell from your secret
manager. The provisioning script generates temporary object-scoped URLs; the
R2 credentials themselves are not sent to Vast.

```bash
python human/vast_start_rank_training.py \
  --dataset-key rustify/human-ranks/20260926-hcpe-vast.tar \
  --result-key rustify/human-ranks/results/20260926-rank-models.tar.gz \
  --label human-rank-training-20260926
```

The default GPU is a verified on-demand Tesla V100, with an hourly price cap of
$0.25, a 40 GB disk, and 2級・初段・三段 training. Change `--gpu`,
`--max-hourly`, `--disk-gb`, `--bands`, or `--epochs` to adjust the run. The
Vast.ai CLI can also be installed with `curl -fsSL https://vast.ai/install.sh | bash`;
use `vastai show instance INSTANCE_ID` to check its state. Fetch its
container logs with:

```bash
VAST_API_TOKEN="$VAST_API_TOKEN" python human/vast_instance_logs.py INSTANCE_ID --tail 100
```

After training, the instance uploads checkpoints, ONNX files, policy evaluation,
and logs to the specified result object.

On the source host, package just the HCPE splits and dataset metadata (this
omits the source KIF/CSA files and any partial checkpoints):

```bash
./human/pack_vast_dataset.sh /home/user/human_data_ranks/20260926 \
  /tmp/human-ranks-20260926.tar.gz
vastai copy local:/tmp/human-ranks-20260926.tar.gz \
  INSTANCE_ID:/workspace/human-ranks-20260926.tar.gz
```

Clone this branch on the Vast instance and run:

```bash
DATA_DIR=/workspace/human_data_ranks/20260926 \
DATA_BUNDLE=/workspace/human-ranks-20260926.tar.gz \
  ./human/vast_rank_train.sh
tail -f /workspace/human_data_ranks/20260926/logs/vast-rank-*.log
```

The runner backgrounds the job and resumes from existing outputs. It defaults
to all nine bands from 3級 through 六段. Each trained checkpoint is exported
to ONNX under `DATA_DIR/onnx/`; `DATA_DIR/eval-policy.tsv` records top-1 move
match on up to 50,000 held-out positions per rank. To train only the original
target ranks (2級, 初段, 三段), set:

```bash
BANDS="0029-0029 0031-0031 0033-0033" \
  DATA_DIR=/workspace/human_data_ranks/20260926 ./human/vast_rank_train.sh
```

The runner expects the HCPE data to be prebuilt and sets
`SKIP_DATA_BUILD=1`. `BAND_POSITIONS`, `EPOCHS`, `BASE_POSITIONS`, `BATCHSIZE`,
`GPU`, `BLOCKS`, `CHANNELS`, and `AMP_DTYPE` can be overridden in the same way.

## 2. Train one model per rating band

```bash
python -m pydlshogi2.train out/1800-2099/train.hcpe out/1800-2099/test.hcpe \
    --gpu 0 --val_lambda 1.0
```

Use `--val_lambda 1.0` (outcome-only value): for imitation the **policy** is
what matters. Evaluate with policy move-matching accuracy (the trainer already
logs `test accuracy`, the fraction of positions where the top policy move equals
the human move) rather than playing strength.

## 3. Play human-like

Play with little or no search so the human-trained policy is not "corrected"
toward superhuman moves — e.g. a single playout / direct policy sampling. (A
dedicated policy-only player can be added later.)
