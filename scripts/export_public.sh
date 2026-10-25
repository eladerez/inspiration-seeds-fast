#!/usr/bin/env bash
# Build a clean public copy of this repo with fresh git history: every tracked file except private planning notes,
# the LinkedIn draft and raw job logs. Scans the copy for secrets and private paths before committing.
# Usage: scripts/export_public.sh [target dir, default ../inspiration-seeds-fast-public]
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="${1:-$SRC/../inspiration-seeds-fast-public}"
EXCLUDE='^(TASK\.md|CLAUDE\.md|docs/linkedin_post\.md|logs/.*\.log)$'

[[ -z "$(git -C "$SRC" status --porcelain)" ]] || { echo "commit or stash changes in $SRC first" >&2; exit 1; }
if [[ -d $DST/.git ]]; then echo "$DST already has a git repo; remove it or pick another target" >&2; exit 1; fi
mkdir -p "$DST"
git -C "$SRC" ls-files | grep -v -E "$EXCLUDE" | rsync -a --files-from=- "$SRC/" "$DST/"

# Fail on anything that looks like a credential or a private path.
if grep -rIn -E 'AIza[0-9A-Za-z_-]{30}|hf_[A-Za-z0-9]{30}|BEGIN (RSA|PRIVATE)|/home/[a-z]+/' "$DST" --exclude-dir=pairs; then
  echo "possible secret or private path above; not committing" >&2; exit 1
fi

cd "$DST"
git init -q -b main
git config user.name "$(git -C "$SRC" config user.name)"
git config user.email "$(git -C "$SRC" config user.email)"
git add -A
git commit -q -m "Inspiration Seeds, Faster: speed vs creativity benchmark

Eight configurations of Inspiration Seeds (FLUX.1 Kontext [dev] + LoRA), all on one RTX PRO 6000, on the paper's full benchmark (99 pairs x 4
seeds): timing, Gemini description length, DreamSim diversity and fidelity, paired statistics, project page and
the 99-pair viewer, and a faster generate.py.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
echo "exported $(git ls-files | wc -l) files to $DST ($(du -sh . | cut -f1))"
