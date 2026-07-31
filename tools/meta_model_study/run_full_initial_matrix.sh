#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
PLAN="${PLAN:-$REPO_ROOT/tools/meta_model_dataset/plan_wide_v2.json}"
BASE_ROOT="${BASE_ROOT:-$REPO_ROOT/data/meta_model/setting_to_loss_wide_v2}"
EXTENDED_ROOT="${EXTENDED_ROOT:-$REPO_ROOT/data/meta_model/setting_to_loss_wide_v2_extended_v1}"
SNAPSHOT="${SNAPSHOT:-$REPO_ROOT/artifacts/wide_v2_full30_gt_snapshot.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/data/meta_model_studies/wide_v2_full30_initial_matrix}"
JOBS="${JOBS:-4}"
TRACK_JOBS="${TRACK_JOBS:-8}"
METHODS="${METHODS:-constant_mean,compact_ridge,extra_trees}"

if [[ -n "${WAIT_FOR_TMUX_SESSION:-}" ]]; then
  while tmux has-session -t "$WAIT_FOR_TMUX_SESSION" 2>/dev/null; do
    sleep 60
  done
fi

mapfile -t experiments < <(
  "$PYTHON" -c '
import json
import sys

plan = json.load(open(sys.argv[1], encoding="utf-8"))
for experiment in plan["experiments"]:
    print(experiment["experiment_id"])
' "$PLAN"
)

freeze_args=(--output "$SNAPSHOT")
for experiment in "${experiments[@]}"; do
  extended="$EXTENDED_ROOT/$experiment"
  base="$BASE_ROOT/$experiment"
  if [[ -f "$extended/rescue_audit.json" ]]; then
    freeze_args+=(--environment "$extended")
  else
    freeze_args+=(--environment "$base")
  fi
done

"$PYTHON" -m tools.meta_model_study.freeze_wide_snapshot "${freeze_args[@]}"

"$PYTHON" -m pytest -q \
  tests/test_meta_model_study_features.py \
  tests/test_meta_model_study_wide.py \
  tests/test_meta_model_study_models.py \
  tests/test_meta_model_study_wide_run.py \
  tests/test_meta_model_study_bounds.py \
  tests/test_meta_model_study_initial_matrix.py

"$PYTHON" -m tools.meta_model_study.bounds \
  --snapshot-manifest "$SNAPSHOT" \
  --split validation \
  > "${SNAPSHOT%.json}_bounds.json"

"$PYTHON" -m tools.meta_model_study.initial_matrix \
  --snapshot-manifest "$SNAPSHOT" \
  --output-root "$OUTPUT_ROOT" \
  --jobs "$JOBS" \
  --track-jobs "$TRACK_JOBS" \
  --methods "$METHODS"
