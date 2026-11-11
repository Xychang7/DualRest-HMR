#!/bin/sh
set -eu
ROOT=/root/sam-body4d-0.2.0
OUT="$ROOT/outputs/dynamesh_v2_phase2_validation"
PIDFILE="$OUT/evaluation.pid"
mkdir -p "$OUT"
if [ -f "$PIDFILE" ]; then
    old_pid=$(cat "$PIDFILE")
    if kill -0 "$old_pid" 2>/dev/null; then
        echo "evaluator already running pid=$old_pid" >&2
        exit 2
    fi
fi
cd "$ROOT"
nohup /root/miniconda3/envs/duomo/bin/python scripts/evaluate_dynamesh_v2_phase2.py > "$OUT/run.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$PIDFILE.tmp"
mv "$PIDFILE.tmp" "$PIDFILE"
echo "$pid"
