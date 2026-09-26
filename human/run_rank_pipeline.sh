#!/bin/bash
# End-to-end: Shogi Wars KIF -> nine rank models (3級 .. 六段), Maia style.
#
#   1. KIF -> CSA (many shards; each is parsed in RAM later)
#   2. CSA -> per-rank HCPE, one band per rank ordinal 28..36
#   3. base model on positions from every rank (capped at BASE_POSITIONS)
#   4. fine-tune the base on each of the nine bands (capped at BAND_POSITIONS),
#      so the thin 四段-六段 bands start from a model that already plays shogi
#
# Every step is skipped when its output already exists, so a re-run resumes.
#
# The KIF corpus keeps growing, so step 1 writes DATA_DIR/kif_manifest.tsv
# (name, size, sha1, status, shard of every KIF read) and DATA_DIR/DATASET.md
# (source, counts, manifest hash). Use a new DATA_DIR for every snapshot.
#
#   DATA_DIR=~/human_data_ranks/20260926 nohup ./human/run_rank_pipeline.sh \
#       > ~/human_data_ranks/20260926.log 2>&1 &
#
# Tunables: KIF_DIR (~/data/kif), DATA_DIR (~/human_data_ranks/latest),
# CSA_DIR ($DATA_DIR/csa), KIF_SOURCE (free text for DATASET.md),
# BASE_POSITIONS (20000000), BAND_POSITIONS (4000000), BASE_EPOCHS (1),
# EPOCHS (2), plus everything train_all_bands.sh takes.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
KIF_DIR="${KIF_DIR:-$HOME/data/kif}"
export DATA_DIR="${DATA_DIR:-$HOME/human_data_ranks/latest}"
CSA_DIR="${CSA_DIR:-$DATA_DIR/csa}"
KIF_SOURCE="${KIF_SOURCE:-$KIF_DIR}"
BASE_POSITIONS="${BASE_POSITIONS:-20000000}"
BAND_POSITIONS="${BAND_POSITIONS:-4000000}"
BASE_EPOCHS="${BASE_EPOCHS:-1}"
export EPOCHS="${EPOCHS:-2}"
export PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python}"
export BLOCKS="${BLOCKS:-10}" CHANNELS="${CHANNELS:-192}" AMP_DTYPE="${AMP_DTYPE:-float16}"
REC=38   # bytes per HuffmanCodedPosAndEval record
BANDS="0028-0028 0029-0029 0030-0030 0031-0031 0032-0032 0033-0033 0034-0034 0035-0035 0036-0036"

cd "$REPO_DIR"
mkdir -p "$DATA_DIR"

if [ ! -s "$CSA_DIR/shogiwars-00.csa" ]; then
    echo "=== KIF -> CSA ($(date '+%F %T')) ==="
    "$PYTHON" human/kif_to_csa.py "$KIF_DIR" "$CSA_DIR" --shards 256 \
        --manifest "$DATA_DIR/kif_manifest.tsv"
fi

if [ ! -s "$DATA_DIR/0028-0028/train.hcpe" ]; then
    echo "=== CSA -> per-rank HCPE ($(date '+%F %T')) ==="
    "$PYTHON" human/csa_to_hcpe_by_rating.py "$CSA_DIR" "$DATA_DIR" \
        --bands 28,29,30,31,32,33,34,35,36,37 --filter_moves 20 | tee "$DATA_DIR/hcpe_counts.txt"
fi

if [ ! -s "$DATA_DIR/DATASET.md" ]; then
    {
        echo "# Shogi Wars rank dataset ($(basename "$DATA_DIR"))"
        echo
        echo "- KIF source: $KIF_SOURCE"
        echo "- KIF dir: \`$KIF_DIR\`"
        echo "- built: $(date '+%F %T %Z') on $(hostname), repo $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
        echo "- manifest: \`kif_manifest.tsv\` (name, size, sha1, status, shard), sha256 $(sha256sum "$DATA_DIR/kif_manifest.tsv" | cut -c1-64)"
        echo "- KIF files by status:"
        tail -n +2 "$DATA_DIR/kif_manifest.tsv" | cut -f4 | sort | uniq -c | awk '{print "  - " $2 ": " $1}'
        echo
        echo "Bands: one per rank ordinal (3級=0028 .. 六段=0036); 0000-0027 and 0037-up are"
        echo "catch-alls used only in the base model. Records per band/split:"
        echo
        echo '```'
        cat "$DATA_DIR/hcpe_counts.txt"
        echo '```'
    } > "$DATA_DIR/DATASET.md"
fi

# 各帯を BAND_POSITIONS に切り詰めた学習ファイル (帯がそれより小さければ全部)
for band in $BANDS; do
    [ -s "$DATA_DIR/$band/train.hcpe" ] || continue
    [ -s "$DATA_DIR/$band/train_capped.hcpe" ] || \
        head -c $((BAND_POSITIONS * REC)) "$DATA_DIR/$band/train.hcpe" > "$DATA_DIR/$band/train_capped.hcpe"
done

# ベースモデル: 全段級 (帯の外も含む) から均等に BASE_POSITIONS を集める
BASE="$DATA_DIR/base"
mkdir -p "$BASE"
if [ ! -s "$BASE/train.hcpe" ]; then
    echo "=== building the base training set ($(date '+%F %T')) ==="
    all=("$DATA_DIR"/*/train.hcpe)
    per=$((BASE_POSITIONS / ${#all[@]}))
    : > "$BASE/train.hcpe"; : > "$BASE/test.hcpe"
    for f in "${all[@]}"; do
        [ "$f" = "$BASE/train.hcpe" ] && continue
        head -c $((per * REC)) "$f" >> "$BASE/train.hcpe"
        t="$(dirname "$f")/test.hcpe"
        [ -s "$t" ] && head -c $((20000 * REC)) "$t" >> "$BASE/test.hcpe"
    done
fi
BASE_MODEL="$BASE/model-base-$(printf '%03d' "$BASE_EPOCHS").pth"
if [ ! -s "$BASE_MODEL" ]; then
    echo "=== training the base model ($(date '+%F %T')) ==="
    "$PYTHON" -m pydlshogi2.train "$BASE/train.hcpe" "$BASE/test.hcpe" \
        --gpu "${GPU:-0}" --amp --amp_dtype "$AMP_DTYPE" --blocks "$BLOCKS" --channels "$CHANNELS" \
        --epoch "$BASE_EPOCHS" --batchsize "${BATCHSIZE:-256}" --lr "${LR:-0.01}" \
        --val_lambda 1.0 --save_interval 2000 \
        --log "$BASE/train-base.log" --checkpoint "$BASE/model-base-{epoch:03}.pth"
fi

# テストが評価バッチ (1024) に満たない帯は学習できないので知らせておく
for band in $BANDS; do
    n=$(( $(stat -c %s "$DATA_DIR/$band/test.hcpe" 2>/dev/null || echo 0) / REC ))
    [ "$n" -ge 1024 ] || echo "WARNING: $band has only $n test positions; it will be skipped or fail"
done

echo "=== fine-tuning the nine rank models ($(date '+%F %T')) ==="
BANDS="$BANDS" INIT_MODEL="$BASE_MODEL" TRAIN_NAME=train_capped.hcpe LR="${FT_LR:-0.005}" \
    "$SCRIPT_DIR/train_all_bands.sh"
echo "pipeline finished ($(date '+%F %T'))"
