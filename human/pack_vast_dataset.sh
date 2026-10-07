#!/usr/bin/env bash
# Package only the prebuilt HCPE splits and their dataset metadata for Vast.ai.
# This deliberately excludes work files and any partial checkpoints in DATA_DIR.
set -euo pipefail

SOURCE_DATA_DIR="${1:-${DATA_DIR:-$HOME/human_data_ranks/latest}}"
OUTPUT="${2:-$HOME/human-ranks-$(date '+%Y%m%d-%H%M%S').tar.gz}"
RANK_DIRS=(
    0000-0027
    0028-0028 0029-0029 0030-0030 0031-0031
    0032-0032 0033-0033 0034-0034 0035-0035 0036-0036
    0037-up
)

[ -d "$SOURCE_DATA_DIR" ] || { echo "missing dataset directory: $SOURCE_DATA_DIR" >&2; exit 1; }
[ -s "$SOURCE_DATA_DIR/DATASET.md" ] || { echo "missing $SOURCE_DATA_DIR/DATASET.md" >&2; exit 1; }
[ -s "$SOURCE_DATA_DIR/hcpe_counts.txt" ] || { echo "missing $SOURCE_DATA_DIR/hcpe_counts.txt" >&2; exit 1; }
[ ! -e "$OUTPUT" ] || { echo "refusing to overwrite $OUTPUT" >&2; exit 1; }

FILES=(DATASET.md hcpe_counts.txt)
for band in "${RANK_DIRS[@]}"; do
    for split in train test; do
        path="$band/$split.hcpe"
        [ -s "$SOURCE_DATA_DIR/$path" ] || { echo "missing $SOURCE_DATA_DIR/$path" >&2; exit 1; }
        FILES+=("$path")
    done
done

mkdir -p "$(dirname "$OUTPUT")"
tar -czf "$OUTPUT" -C "$SOURCE_DATA_DIR" "${FILES[@]}"
echo "Created $OUTPUT"
du -h "$OUTPUT"
