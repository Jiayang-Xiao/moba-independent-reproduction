#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
PY=${MOBA_PYTHON:-$ROOT/.venv/bin/python}
SOURCE=${MOBA_SOURCE:-/home/xjy/strict_zs_cross_domain_vad/third_party/MPN/data/shanghai/training/frames}
TARGET=${MOBA_TARGET_ROOT:-/home/xjy/strict_zs_cross_domain_vad/third_party/MPN/data}
OUT=${MOBA_OUTPUT:-$ROOT/outputs/flow_v1_seed17}
CACHE=${MOBA_CACHE:-$ROOT/.private/source_flow}
WEIGHTS=${MOBA_FLOW_CHECKPOINT:-$ROOT/.private/weights/FlowNet2_checkpoint.pth.tar}
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${MOBA_GPU:-0}
mkdir -p -- "$OUT"
exec 9>"$OUT/run.lock"
flock -n 9 || { echo 'This MoBA output already has an active run' >&2; exit 1; }
exec >>"$OUT/execution.log" 2>&1
PHASE=PREPARING
on_exit() {
  status=$?
  printf '%s\n' "$status" > "$OUT/last_exit.txt"
  if [[ "$status" != 0 ]]; then
    if [[ "$PHASE" == PUBLISHING ]]; then
      "$PY" "$ROOT/scripts/state.py" "$OUT/state.json" COMPLETED_UPLOAD_PENDING "$status" || true
    else
      "$PY" "$ROOT/scripts/state.py" "$OUT/state.json" FAILED "$status" || true
    fi
    echo "Stopped in phase $PHASE; inspect execution.log; saved checkpoints/cache retained."
  fi
}
trap on_exit EXIT
exec 8>"${TMPDIR:-/tmp}/xjy-moba-gpu-${CUDA_VISIBLE_DEVICES//[^a-zA-Z0-9_-]/_}.lock"
echo "Waiting for MoBA GPU lock: $CUDA_VISIBLE_DEVICES"
flock 8
cd -- "$ROOT"
"$PY" scripts/verify_release.py
"$PY" scripts/state.py "$OUT/state.json" PREPARING
"$PY" src/test_local.py --full --device cuda:0 --output "$OUT/local_checks_server.json"
"$PY" src/prepare.py --source "$SOURCE" --target-root "$TARGET" --out "$OUT" --checkpoint "$WEIGHTS" --device cuda:0
PHASE=CACHING_SOURCE_FLOW
"$PY" scripts/state.py "$OUT/state.json" "$PHASE"
"$PY" src/cache_flow.py --source "$SOURCE" --cache "$CACHE" --out "$OUT" --checkpoint "$WEIGHTS" --device cuda:0
PHASE=TRAINING
"$PY" scripts/state.py "$OUT/state.json" "$PHASE"
"$PY" src/train.py --source "$SOURCE" --cache "$CACHE" --out "$OUT" --checkpoint "$WEIGHTS" --device cuda:0
PHASE=EVALUATING
"$PY" scripts/state.py "$OUT/state.json" "$PHASE"
"$PY" src/evaluate.py --source "$SOURCE" --target-root "$TARGET" --out "$OUT" --device cuda:0
PHASE=PUBLISHING
"$PY" scripts/state.py "$OUT/state.json" "$PHASE"
bash scripts/publish.sh
echo "Completed source training, both RGB-only targets and GitHub publication: $OUT/summary.json"
