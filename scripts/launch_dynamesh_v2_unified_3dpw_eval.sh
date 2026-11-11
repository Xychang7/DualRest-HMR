#!/usr/bin/env bash
set -euo pipefail

project=/root/sam-body4d-0.2.0
output="$project/outputs/dynamesh_v2_unified_3dpw_eval"
python=/root/miniconda3/envs/duomo/bin/python
script="$project/scripts/evaluate_dynamesh_v2_unified_3dpw.py"

mkdir -p "$output"
if pgrep -af '[e]valuate_dynamesh_v2_unified_3dpw.py --run' >/dev/null; then
  echo "Protocol B evaluator is already running" >&2
  exit 1
fi

cd "$project"
nohup "$python" "$script" --run >"$output/run.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$output/evaluation.pid"
sleep 2
kill -0 "$pid"
printf 'pid=%s\n' "$pid"
