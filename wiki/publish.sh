#!/bin/bash
# Publish wiki/*.md to the GitHub wiki.
#
# The wiki is a separate git repository. This clones it, mirrors the Markdown
# files from this directory into it, and pushes if anything changed.
#
# Usage:
#   ./wiki/publish.sh              # publish
#   ./wiki/publish.sh --dry-run    # show the diff without pushing
#
# Note: a GitHub wiki repository does not exist until the wiki has at least one
# page. If the clone below fails on a fresh repository, create any page once in
# the browser (it will be overwritten) and re-run.
set -e

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
fi

SRC="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SRC/.." && pwd)"
WIKI_URL="${WIKI_URL:-$(git -C "$REPO_ROOT" remote get-url origin | sed 's/\.git$//').wiki.git}"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

echo "cloning $WIKI_URL"
if ! git clone --quiet "$WIKI_URL" "$TMPDIR/wiki" 2>/dev/null; then
    echo "error: could not clone the wiki repository." >&2
    echo "       A GitHub wiki has no git repository until its first page exists." >&2
    echo "       Create one page at ${WIKI_URL%.wiki.git}/wiki and re-run." >&2
    exit 1
fi

# 既存ページを一旦削除してから配置し直す (ここが唯一の情報源)
find "$TMPDIR/wiki" -maxdepth 1 -name '*.md' -delete
cp "$SRC"/*.md "$TMPDIR/wiki/"
# wiki/README.md はソース側の運用説明なのでwikiには載せない
rm -f "$TMPDIR/wiki/README.md"

cd "$TMPDIR/wiki"
if git diff --quiet && [ -z "$(git status --porcelain)" ]; then
    echo "wiki is already up to date"
    exit 0
fi

git add -A
git --no-pager diff --cached --stat

if [ "$DRY_RUN" = 1 ]; then
    echo "(dry run; nothing pushed)"
    exit 0
fi

git commit --quiet -m "Sync wiki from $(git -C "$REPO_ROOT" rev-parse --short HEAD)"
git push --quiet
echo "published to ${WIKI_URL%.wiki.git}/wiki"
