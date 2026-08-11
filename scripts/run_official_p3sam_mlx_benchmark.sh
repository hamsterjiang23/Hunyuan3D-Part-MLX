#!/bin/sh
set -eu

output="artifacts/official_paper_protocol_mlx_200_float64_batch1"
mkdir -p "$output"

attempt=1
max_attempts=5
while [ "$attempt" -le "$max_attempts" ]; do
  if PYTHONUNBUFFERED=1 .venv/bin/python scripts/benchmark_p3sam_partobjaverse.py \
    --backend mlx \
    --dataset models/PartObjaverse-Tiny \
    --weights models/p3sam.safetensors \
    --output "$output" \
    --points 100000 \
    --prompts 400 \
    --prompt-batch-size 1 \
    --seed 42 \
    --official-fps-start \
    --no-clean-mesh \
    --postprocess \
    --postprocess-threshold 0.95; then
    break
  else
    status=$?
  fi
  echo "benchmark worker exited with status $status; resume attempt $attempt/$max_attempts" >&2
  if [ "$attempt" -eq "$max_attempts" ]; then
    exit "$status"
  fi
  attempt=$((attempt + 1))
  sleep 5
done

.venv/bin/python scripts/summarize_p3sam_benchmark.py \
  --records "$output/records.jsonl" \
  --metadata models/PartObjaverse-Tiny/PartObjaverse-Tiny_semantic.json \
  --output "$output/paper_summary.json"
