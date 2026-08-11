#!/bin/sh
set -eu

output="artifacts/official_paper_oracle_mlx_200_float64_batch1"
mkdir -p "$output"

attempt=1
max_attempts=5
while [ "$attempt" -le "$max_attempts" ]; do
  if PYTHONUNBUFFERED=1 .venv/bin/python scripts/benchmark_p3sam_paper_oracle.py \
    --backend mlx \
    --dataset models/PartObjaverse-Tiny \
    --weights models/p3sam.safetensors \
    --output "$output" \
    --points 100000 \
    --interactive-prompts-per-part 10 \
    --prompt-batch-size 1 \
    --seed 42 \
    --postprocess-threshold 0.95; then
    exit 0
  else
    status=$?
  fi
  printf 'paper oracle attempt %s/%s exited %s; resuming\n' "$attempt" "$max_attempts" "$status" >&2
  attempt=$((attempt + 1))
done

printf 'paper oracle failed after %s attempts\n' "$max_attempts" >&2
exit 1
