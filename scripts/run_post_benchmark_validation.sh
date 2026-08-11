#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

BENCHMARK_PID_FILE=${1:-artifacts/official_paper_protocol_mlx_200_batch1/benchmark.pid}
if [ -f "$BENCHMARK_PID_FILE" ]; then
  BENCHMARK_PID=$(cat "$BENCHMARK_PID_FILE")
  while kill -0 "$BENCHMARK_PID" 2>/dev/null; do
    sleep 30
  done
fi

PYTHONPATH=src .venv/bin/python scripts/recompute_p3sam_benchmark_metrics.py \
  --benchmark artifacts/official_paper_protocol_mlx_200_batch1 \
  --dataset models/PartObjaverse-Tiny

PYTHONPATH=src .venv/bin/python scripts/summarize_p3sam_benchmark.py \
  --records artifacts/official_paper_protocol_mlx_200_batch1/records.jsonl \
  --metadata models/PartObjaverse-Tiny/PartObjaverse-Tiny_semantic.json \
  --output artifacts/official_paper_protocol_mlx_200_batch1/paper_summary.json

SAMPLE_UID=00200996b8f34f55a2dd2f44d316d107
MESH="models/PartObjaverse-Tiny/PartObjaverse-Tiny_mesh/$SAMPLE_UID.glb"
TARGET="models/PartObjaverse-Tiny/PartObjaverse-Tiny_instance_gt/$SAMPLE_UID.npy"
REPLAY="artifacts/fixed_replay_002/replay_manifest.npz"
MLX_RESULT="artifacts/fixed_replay_002/mlx"

PYTHONPATH=src .venv/bin/python scripts/run_p3sam_mlx.py "$MESH" \
  --weights models/p3sam.safetensors \
  --output "$MLX_RESULT" \
  --points 100000 \
  --prompts 400 \
  --prompt-batch-size 1 \
  --seed 42 \
  --official-fps-start \
  --no-clean-mesh \
  --connectivity \
  --postprocess \
  --postprocess-threshold 0.95 \
  --replay-manifest "$REPLAY" \
  --trace-dir "$MLX_RESULT" \
  --trace-full-tensors

PYTHONPATH=src .venv/bin/python scripts/evaluate_p3sam_stages.py \
  --result "$MLX_RESULT" \
  --target "$TARGET" \
  --output "$MLX_RESULT/stage_metrics.json"

BALUSTRADE_RESULT="artifacts/official_full_balustrade"
PYTHONPATH=src .venv/bin/python scripts/run_p3sam_mlx.py \
  inputs/curved-lantern-balustrade-final.glb \
  --weights models/p3sam.safetensors \
  --output "$BALUSTRADE_RESULT" \
  --points 100000 \
  --prompts 400 \
  --prompt-batch-size 1 \
  --seed 42 \
  --official-fps-start \
  --clean-mesh \
  --connectivity \
  --postprocess \
  --postprocess-threshold 0.95 \
  --trace-dir artifacts/official_full_balustrade_trace

PYTHONPATH=src .venv/bin/python scripts/analyze_p3sam_parts.py \
  "$BALUSTRADE_RESULT/segmented.glb" \
  --labels "$BALUSTRADE_RESULT/face_ids.npy" \
  --output "$BALUSTRADE_RESULT/part_analysis.json"
