# Task Costs & I/O Dependencies

Measured resource costs and data dependencies for Hunyuan3D-Part validation.

## Cost Lookup Table

| Task | Category | Runtime | Memory | Ext. Resources | Notes |
|---|---|---|---|---|---|
| P3-SAM MLX, one PartObj sample | Moderate | ~35-38 s | 3.53 GB peak | Apple GPU | 100K points, 400 prompts, batch 1, official float64 preprocessing and full postprocess |
| P3-SAM MLX, PartObj-Tiny 200 | Heavy | ~2.0 h | 3.53 GB peak | Apple GPU | Resume-safe JSONL, one worker, projected/connectivity/full metrics |
| P3-SAM CUDA, one PartObj sample | Heavy | ~30-33 s | 8.15 GB allocated | RTX 4070 12 GB | Prompt batch 8; batch 32 causes memory pressure on this GPU |
| X-Part MLX, 50-step generation | Heavy | ~230 s | 29.49 GB | Apple GPU | 10 parts, 82K points, resolution 128 |
| ShapeVAE MLX decode, resolution 128 | Moderate | ~5 s | 2.72 GB | Apple GPU | 10 saved part latents |
| ShapeVAE MLX decode, resolution 512 | Heavy | Unknown; probe required | Unknown | Apple GPU | Run after P3 full benchmark to avoid GPU contention |

Any 200-sample P3 batch is Heavy because aggregate runtime exceeds one hour.

## I/O Dependency Table

| Task | Reads | Writes |
|---|---|---|
| P3-SAM MLX full benchmark | `models/PartObjaverse-Tiny`, `models/p3sam.safetensors` | `artifacts/official_paper_protocol_mlx_200_float64_batch1` |
| P3-SAM CUDA benchmark | local PartObj dataset and P3 weight | `artifacts/official_paper_protocol_cuda_200_float64_bs8` |
| Fixed CUDA/MLX trace | one immutable PartObj mesh, GT and replay manifest | `artifacts/fixed_replay_002_float64/{cuda,mlx}` |
| X-Part full generation | input GLB and all Hunyuan3D-Part weights | `artifacts/xpart_fullscale_steps50_r128` |
| ShapeVAE resolution-512 decode | input GLB, ShapeVAE weight, saved `latents.npy` | `artifacts/xpart_fullscale_steps50_r512` |
| P3 benchmark summarizer | `records.jsonl`, semantic metadata | a separate `paper_summary.json` |

The resolution-512 decode reads immutable saved latents and writes a new output directory, so it does not conflict with the completed resolution-128 output. It should not overlap the active P3 MLX benchmark because both use the Apple GPU.
