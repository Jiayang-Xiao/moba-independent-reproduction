#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
PY=${MOBA_PYTHON:-$ROOT/.venv/bin/python}
OUT=${MOBA_OUTPUT:-$ROOT/outputs/flow_v1_seed17}
REPO=Jiayang-Xiao/moba-independent-reproduction
TAG=moba-independent-v1-seed17
cd -- "$ROOT"
"$PY" scripts/verify_release.py
"$PY" scripts/export_results.py --out "$OUT"
source "$ROOT/scripts/github_env.sh"
if [[ ! -d .git ]]; then git init; fi
[[ $(git rev-parse --show-toplevel) == "$ROOT" ]] || { echo 'Refusing a parent repository' >&2; exit 1; }
if ! git rev-parse --verify HEAD >/dev/null 2>&1; then git symbolic-ref HEAD refs/heads/main; fi
[[ $(git symbolic-ref -q HEAD) == refs/heads/main ]] || { echo 'Publication requires main' >&2; exit 1; }
if ! git config user.name >/dev/null; then git config user.name Jiayang-Xiao; fi
if ! git config user.email >/dev/null; then git config user.email 'Jiayang-Xiao@users.noreply.github.com'; fi
if ! git diff --cached --quiet; then echo 'Unexpected staged files; publication stopped' >&2; exit 1; fi
git add -- README.md .gitignore config.json release_manifest.json src scripts docs third_party evidence
if ! git diff --cached --quiet; then git commit -m 'Publish independent MoBA flow candidate and completed source-only reproduction evidence'; fi
REMOTE="git@github.com:$REPO.git"
if gh api "repos/$REPO" >/dev/null 2>&1; then
  [[ $(gh api "repos/$REPO" --jq .visibility) == public ]] || { echo 'Named repository is not public; refusing visibility change' >&2; exit 1; }
  if ! git config --get remote.origin.url >/dev/null; then
    HEADS=$(git ls-remote "$REMOTE" refs/heads/main)
    [[ -z "$HEADS" ]] || { echo 'Remote already has main; refusing to adopt an unrelated repository' >&2; exit 1; }
  fi
else
  gh repo create "$REPO" --public --description 'Independent paper-guided MoBA reproduction: fixed source-only training, RGB-only cross-domain AUROC and disclosed assumptions'
fi
if git config --get remote.origin.url >/dev/null; then
  [[ $(git config --get remote.origin.url) == "$REMOTE" ]] || { echo 'Unexpected origin' >&2; exit 1; }
else
  git remote add origin "$REMOTE"
fi
git push -u origin main
COMMIT=$(git rev-parse HEAD)
REMOTE_COMMIT=$(git ls-remote origin refs/heads/main | awk '{print $1}')
[[ "$COMMIT" == "$REMOTE_COMMIT" ]] || { echo 'Remote main does not match local commit' >&2; exit 1; }
if ! gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
  gh release create "$TAG" --repo "$REPO" --target "$COMMIT" \
    --title 'Independent MoBA V1: seed17, final source checkpoint' --notes-file docs/release_notes.md
fi
RELEASE_TARGET=$(gh api "repos/$REPO/releases/tags/$TAG" --jq .target_commitish)
[[ "$RELEASE_TARGET" == "$COMMIT" ]] || { echo 'Existing release targets a different commit; refusing asset replacement' >&2; exit 1; }
# Idempotent asset upload; no FlowNet2 teacher or dataset is published.
gh release upload "$TAG" "$ROOT/.private/publication/model.pt" "$ROOT/.private/publication/review_bundle.zip" \
  --repo "$REPO" --clobber
"$PY" scripts/state.py "$OUT/state.json" PUBLISHED 0
printf '0\n' > "$OUT/last_exit.txt"
printf 'Repository: https://github.com/%s\nBranch: main\nCommit: %s\nRelease: https://github.com/%s/releases/tag/%s\n' "$REPO" "$COMMIT" "$REPO" "$TAG"
