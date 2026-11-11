#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/sam-body4d-0.2.0
OUT=$ROOT/outputs/dynamesh_v2_camera_phase2
mkdir -p "$OUT"
cd "$ROOT"
exec /root/miniconda3/envs/duomo/bin/python -u scripts/train_dynamesh_v2_camera_phase2.py
