#!/usr/bin/env bash
# Unattended Vast.ai entrypoint. DATA_BUNDLE_URL and RESULT_UPLOAD_URL should
# be short-lived, object-scoped S3 presigned URLs.
set -Eeuo pipefail

WORK_ROOT="${WORK_ROOT:-/workspace}"
DATA_DIR="${DATA_DIR:-$WORK_ROOT/human_data_ranks/20260926}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/python-dlshogi2-human-like}"
BRANCH="${BRANCH:-human-like}"
DATA_BUNDLE_URL="${DATA_BUNDLE_URL:?set a presigned GET URL for the HCPE archive}"
RESULT_UPLOAD_URL="${RESULT_UPLOAD_URL:?set a presigned PUT URL for the result archive}"
RESULT_ARCHIVE="${RESULT_ARCHIVE:-$WORK_ROOT/human-ranks-results.tar.gz}"

mkdir -p "$WORK_ROOT" "$DATA_DIR"
exec > >(tee -a "$WORK_ROOT/vast-cloud-train.log") 2>&1

upload_results() {
    local rc=$?
    trap - EXIT
    echo "Training process exit status: $rc"
    (cd "$DATA_DIR" && find . -type f \
        \( -name '*.pth' -o -name '*.onnx' -o -name '*.onnx.data' \
        -o -name 'DATASET.md' -o -name 'hcpe_counts.txt' \
        -o -name 'eval-policy.tsv' -o -path './logs/*' \) -print0 \
        | tar --null -czf "$RESULT_ARCHIVE" -C "$DATA_DIR" --files-from -) || true
    if [ -s "$RESULT_ARCHIVE" ]; then
        echo "Uploading result archive ($(stat -c%s "$RESULT_ARCHIVE") bytes)."
        curl --fail --show-error --retry 4 --retry-all-errors \
            --connect-timeout 30 --max-time 7200 \
            --upload-file "$RESULT_ARCHIVE" "$RESULT_UPLOAD_URL"
        echo "Result archive uploaded."
    else
        echo "No result archive was produced." >&2
    fi
    exit "$rc"
}
trap upload_results EXIT

if [ ! -d "$REPO_DIR/.git" ]; then
    git clone --depth 1 --branch "$BRANCH" \
        https://github.com/LoveKapibarasan/python-dlshogi2.git "$REPO_DIR"
fi

if [ ! -s "$DATA_DIR/0029-0029/train.hcpe" ]; then
    bundle="$WORK_ROOT/human-ranks-20260926.tar"
    echo "Downloading HCPE dataset bundle."
    curl --fail --show-error --retry 4 --retry-all-errors \
        --connect-timeout 30 --max-time 7200 -o "$bundle" "$DATA_BUNDLE_URL"
    tar -xf "$bundle" -C "$DATA_DIR"
    rm -f "$bundle"
fi

cd "$REPO_DIR"
if [ ! -x "$REPO_DIR/.venv/bin/python" ]; then
    ./vast_setup.sh
else
    echo "Using existing virtual environment at $REPO_DIR/.venv"
fi

# Focus first on the requested 2級, 初段, 三段 models. Set BANDS at instance
# launch time to include additional ranks when desired.
export DATA_DIR BANDS="${BANDS:-0029-0029 0031-0031 0033-0033}"
export VAST_RANK_TRAIN_FOREGROUND=1
export SKIP_DATA_BUILD=1
export BASE_EPOCHS="${BASE_EPOCHS:-1}"
export EPOCHS="${EPOCHS:-2}"
export BASE_POSITIONS="${BASE_POSITIONS:-20000000}"
export BAND_POSITIONS="${BAND_POSITIONS:-4000000}"
export BATCHSIZE="${BATCHSIZE:-256}"
export BLOCKS="${BLOCKS:-10}"
export CHANNELS="${CHANNELS:-192}"
export AMP_DTYPE="${AMP_DTYPE:-float16}"
export FT_LR="${FT_LR:-0.005}"

echo "Starting rank training for: $BANDS"
./human/vast_rank_train.sh
