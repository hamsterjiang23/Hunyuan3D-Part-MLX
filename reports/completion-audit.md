# Hunyuan3D-Part MLX 完成度审计

日期：2026-08-11。[verified: 本次执行环境日期]

| 原始要求 / 验收项 | 状态 | 权威证据 |
|---|---|---|
| 完整移植公开 P3-SAM 神经网络到 MLX | PASS | [verified: `src/split3d/hunyuan/{sonata_mlx,p3sam_mlx}.py`；固定 replay 可执行] |
| 完整移植公开 `auto_mask.py` topology/postprocess | PASS | [verified: `src/split3d/hunyuan/p3sam_official_postprocess.py`；pinned official parity tests] |
| 使用源 `face_idx` 投影而不是 KD-tree 近似 | PASS | [verified: `project_sample_labels_to_faces` 与 pinned vote test] |
| 邻接 flood-fill、连通区域、漏分区补齐、小部件合并 | PASS | [verified: `tests/test_hunyuan_p3sam_official_postprocess.py` 逐函数官方对照] |
| 官方 mesh cleaning、seeded FPS、connectivity、threshold 0.95 默认参数 | PASS | [verified: `scripts/run_p3sam_mlx.py`、`src/split3d/hunyuan/service.py` defaults] |
| CUDA/MLX 相同 points/normals/face_idx/prompts 逐阶段对齐 | PASS（固定样本） | [verified: `artifacts/fixed_replay_002_float64/{cuda,mlx}`] |
| 同 raw point labels 时 topology face labels 与官方逐面一致 | PASS | [verified: pinned official postprocess tests] |
| 纹理网格的最终与 part GLB 真实写入 face colors | PASS | [verified: `test_save_segmentation_converts_textured_full_and_part_meshes_to_colors`] |
| 完整公开 X-Part：Conditioner → PartFormer → ShapeVAE | PASS | [verified: MLX modules；50-step `xpart_corrected_steps50_r128/runtime.json`] |
| X-Part 论文 full checkpoint | NOT AVAILABLE | [verified: 官方仅公开 light version；`.upstream/hunyuan3d-part/README.md:39-40`] |
| 按论文指标比较 MLX 与 CUDA | PARTIAL | [verified: 固定 replay、5-sample attention A/B、147-sample MLX prefix；论文 GT-prompt evaluator 未公开] |
| 200 样本新版评测 | USER-CANCELLED | [verified: 用户要求“不要进行大规模测试了”；进程停止于 147/200] |
| 真实栏杆模型分割、面积、组件和独立 GLB | PASS | [verified: `reports/balustrade-service-validation.json` 与预览图] |
| 外部 API、`/docs`、OpenAPI、真实 submit/status/download/bundle | PASS | [verified: task `75bb60f5-...`；8080/8083 docs HTTP 200] |
| bundle 递归包含 `parts/*.glb` | PASS（本地修复） | [verified: 新 bundle 清单；service regression test] |
| 最终中文移植与精度报告 | PASS（本地） | [verified: `reports/Hunyuan3D-Part-MLX-port-report.md`] |
| 最新修复、报告和预览同步 Mac | PASS | [verified: 用户明确授权；Mac SHA-256 与本地清单逐字一致] |

## 已同步文件 SHA-256

```text
fa1572676a59ad85d6655b95f9474e1e7b976342cb964c54b28944a8f1e2f970  README.md
2a925d8a6b85911d5ba5fc7045a196e5448056475a120cc7f772258e27eba740  reports/Hunyuan3D-Part-MLX-port-report.md
25e33c9343fbf79f1f30136a8a8ed7b698be697a63bc6916f6923d92ea36020d  reports/balustrade-service-validation.json
62fb77896d3a5daaebdde462f0d4706041ef4ac5b263587d275b515d6bc80a60  reports/assets/balustrade-p3sam-mlx-preview.png
de4e274849177c0bcea18b839360ec556994451d7c857d87c8ec4ffc16bce9ca  scripts/render_glb_preview.py
79f9794fc93e3c42a22252b2c65754266db0ac9f54d09a86ebbcbfdd258978f2  src/split3d/hunyuan/service.py
c5a9125359a5297de05d0b89e204d801809997a3ee4256980bab5a44ec033b23  tests/test_hunyuan_service.py
```

Windows 轻量验证为 17 个 P3-SAM/service tests、3 个 queue tests、Ruff、JSON parse 和 `git diff --check` 全部通过。[verified: 本轮 Windows tool outputs]

Mac 轻量验证为 16 passed、4 skipped；4 个 skip 的原因是 Mac 仓库没有 pinned upstream checkout，相同 pinned parity tests 已在 Windows checkout 执行通过。[verified: Mac pytest `-rs` 与 Windows 17-test run]
