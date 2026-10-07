#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
SESSION=${MOBA_SESSION:-moba-repro-v1}
PY=${MOBA_PYTHON:-$ROOT/.venv/bin/python}
OUT=${MOBA_OUTPUT:-$ROOT/outputs/flow_v1_seed17}
command -v tmux >/dev/null || { echo 'tmux missing; ask server administrator to install it' >&2; exit 1; }
command -v flock >/dev/null || { echo 'flock missing' >&2; exit 1; }
if tmux has-session -t "=$SESSION" 2>/dev/null; then echo "Already running: $SESSION"; exit 0; fi
[[ -x "$PY" ]] || { echo "Python environment missing: $PY" >&2; exit 1; }
"$PY" "$ROOT/scripts/verify_release.py"
mkdir -p -- "$OUT"
printf -v COMMAND '%q ' env \
  "MOBA_AUTH_INTERACTIVE=0" \
  "MOBA_PYTHON=$PY" "MOBA_OUTPUT=$OUT" "MOBA_GPU=${MOBA_GPU:-0}" \
  "MOBA_SOURCE=${MOBA_SOURCE:-/home/xjy/strict_zs_cross_domain_vad/third_party/MPN/data/shanghai/training/frames}" \
  "MOBA_TARGET_ROOT=${MOBA_TARGET_ROOT:-/home/xjy/strict_zs_cross_domain_vad/third_party/MPN/data}" \
  "MOBA_CACHE=${MOBA_CACHE:-$ROOT/.private/source_flow}" \
  "MOBA_FLOW_CHECKPOINT=${MOBA_FLOW_CHECKPOINT:-$ROOT/.private/weights/FlowNet2_checkpoint.pth.tar}" \
  bash "$ROOT/scripts/run.sh"
tmux new-session -d -s "$SESSION" "$COMMAND"
echo "Started tmux session: $SESSION"
echo "Log: $OUT/execution.log"
echo "State: $OUT/state.json"
echo 'After confirming the session and log, SSH may be disconnected. Detach with Ctrl+b, then d; do not press Ctrl+c.'
