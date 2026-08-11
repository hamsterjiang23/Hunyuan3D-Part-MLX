#!/bin/sh
set -eu

output="artifacts/official_paper_protocol_mlx_200_float64_batch1"
mkdir -p "$output"

PYTHONUNBUFFERED=1 .venv/bin/python scripts/benchmark_p3sam_partobjaverse.py \
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
  --postprocess-threshold 0.95

.venv/bin/python scripts/summarize_p3sam_benchmark.py \
  --records "$output/records.jsonl" \
  --metadata models/PartObjaverse-Tiny/PartObjaverse-Tiny_semantic.json \
  --output "$output/paper_summary.json"
