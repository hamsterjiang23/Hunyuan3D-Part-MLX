# Hunyuan3D-Part 原生 MLX 移植与 CUDA 对照最终报告

日期：2026-08-11。[verified: 本次评测会话日期]

目标主机：`mt@10.20.134.22`，Apple M3 Ultra。[verified: Mac `system_profiler` 检查输出]

部署目录：`/Users/mt/hamster/3D_Split`。[verified: SSH 工作目录]

## 1. 最终结论

Hunyuan3D-Part 公开版的完整链路已经移植为原生 MLX：P3-SAM 从 mesh 产生真实面实例分割与包围盒，X-Part 完成条件编码、PartFormer 50 步流匹配、ShapeVAE 逐 part 解码并导出多 geometry GLB。[verified: `artifacts/xpart_corrected_steps50_r128/runtime.json` 中 `part_count=10`；`src/split3d/hunyuan/` 实现]

移植在工程上可运行，但没有复现论文精度。[verified: `artifacts/partobj_full_mlx_corrected/paper_summary.json` 中 `paper_macro_instance_miou=40.4279956415`] MLX 在 PartObjaverse-Tiny 200/200 样本上的论文类别宏平均 mIoU 为 **40.43%**，低于 P3-SAM 论文无 connectivity 的 **59.88%**，差 **19.45 个百分点**。[verified: `artifacts/partobj_full_mlx_corrected/paper_summary.json`；`.upstream/papers/p3-sam.txt:324`]

在相同 7 个可完成 CUDA 样本上，CUDA/MLX 平均 mIoU 为 **33.39% / 33.52%**，均值接近；MLX 平均耗时为 CUDA 的 **1.287×**，平均峰值内存为 **2.695×**。[verified: `artifacts/p3sam_corrected_cuda_mlx_selected7_comparison.json` 中 `cuda` 与 `mlx`] 最终面分区并不等价：7 样本平均 ARI 0.7627、NMI 0.8232、对称最佳 mask IoU 0.7384。[verified: `artifacts/p3sam_corrected_cuda_mlx_selected7_comparison.json`]

修正后的 X-Part 50 步全链路耗时 **230.39 秒**，峰值内存 **29.49 GB**；保存的 latents 可在 **48.26 秒**内按官方默认 resolution 512 解码出 10 个 part。[verified: `artifacts/xpart_corrected_steps50_r128/runtime.json`；`artifacts/xpart_corrected_steps50_r512/decode_runtime.json`] 512 输出包含 1,375,230 顶点和 2,624,969 面，无 NaN/Inf 顶点，但并非所有 part 都 watertight。[verified: `artifacts/xpart_corrected_steps50_r512/surface_metrics.json`]

最终判定：这套 MLX 部署适合研究、可视化和继续优化，不应被标记为论文精度复现或 production-ready watertight 资产流水线。[opinion]

## 2. 环境与权重

| 后端 | 主机与运行时 | 内存 | 证据 |
|---|---|---:|---|
| MLX | Apple M3 Ultra，32 CPU 核、80 GPU 核；macOS 26.4，Python 3.12.13，MLX 0.32.0 | 512 GB 统一内存 | [verified: Mac `system_profiler`、`sw_vers`、运行时版本检查] |
| CUDA | Intel i7-14700KF，RTX 4070；PyTorch 2.8.0+cu128，CUDA 12.8，spconv 2.3.8 | 64 GB RAM，12,282 MiB VRAM | [verified: Windows `nvidia-smi` 与 Python 包版本检查] |

| 权重 | 参数量 | 文件字节数 | 证据 |
|---|---:|---:|---|
| P3-SAM | 112,728,649 | 450,968,044 | [verified: safetensors manifest 检查与 `models/Hunyuan3D-Part/p3sam/p3sam.safetensors`] |
| PartFormer | 3,315,236,672 | 6,630,571,584 | [verified: safetensors manifest 检查与 `models/Hunyuan3D-Part/model/model.safetensors`] |
| Conditioner | 768,258,114 | 1,755,920,004 | [verified: safetensors manifest 检查与 `models/Hunyuan3D-Part/conditioner/conditioner.safetensors`] |
| ShapeVAE | 327,746,177 | 655,539,474 | [verified: safetensors manifest 检查与 `models/Hunyuan3D-Part/shapevae/shapevae.safetensors`] |

四份真实 safetensors 合计 9,492,999,106 字节，并均以严格键名/shape 加载。[verified: 上表四个本地文件大小与模型严格加载运行]

## 3. 移植内容

| 官方模块 | 原生 MLX 实现 | 验证 |
|---|---|---|
| Sonata PointTransformerV3 | `sonata_data.py`、`sonata_mlx.py`、`sparse.py` | [verified: 20K 点 CUDA/MLX feature 对照产物] |
| P3-SAM mask 与 IoU heads | `p3sam_mlx.py` | [verified: 200 个真实 mesh 全量运行] |
| 自动 mask 选择、NMS、bbox | `p3sam_pipeline.py` | [verified: `tests/test_hunyuan_p3sam_pipeline.py` 与端到端产物] |
| X-Part object/part conditioner | `xpart_conditioner_mlx.py` | [verified: 82K 表面点、4096 condition tokens 全链路运行] |
| PartFormer DiT + MoE | `xpart_partformer_mlx.py` | [verified: B=1/B=4 CUDA/MLX 数值产物] |
| ShapeVAE 编解码 | `xpart_shape_mlx.py` | [verified: encoder、latent、SDF 与 resolution 512 解码产物] |
| 50 步 flow 与 GLB 导出 | `xpart_pipeline_mlx.py` | [verified: `artifacts/xpart_corrected_steps50_r128/runtime.json`] |

MLX 神经网络路径没有导入 PyTorch 或调用 CUDA；几何采样、KD-tree、Marching Cubes 和 GLB I/O 使用 NumPy、SciPy、scikit-image 与 trimesh。[verified: `rg "torch|cuda" src/split3d/hunyuan` 仅命中文档字符串；`rg "mlx.core|mlx.nn"` 命中 MLX 模块]

GridSample 已修正为官方 NumPy 默认 quicksort；早期 stable sort 会改变 FNV hash tie 中的代表点，曾产生虚假的 99.998% 面标签一致率，因此旧结果未纳入最终结论。[verified: `src/split3d/hunyuan/sonata_data.py` 与官方 transform 输入逐字段对照] 官方 demo 在 GridSample 消耗 RNG 后再随机选择 FPS 起点；`official_fps_start_index` 已复现该顺序，固定起点 0 仅用于跨后端受控对照。[verified: `tests/test_hunyuan_sonata_data.py` 与真实 20K 点起点索引 543]

## 4. 论文评测协议

P3-SAM 指标对每个 GT part 独立寻找最佳预测 mask IoU，对 shape 内 GT parts 求均值，再先按 8 类分别平均、最后做类别宏平均。[verified: `.upstream/PartField/compute_metric.py:19-28,36-92`] 论文在 PartObj-Tiny 报告无 connectivity 59.88、connectivity 81.14、交互式 51.23。[verified: `.upstream/papers/p3-sam.txt:324,327,330`]

本报告的 P3 全量配置为 200 个 PartObjaverse-Tiny mesh、100,000 表面点、400 prompts、prompt batch 8、seed 42、无 connectivity、固定 FPS 起点 0。[verified: `artifacts/partobj_full_mlx_corrected/records.jsonl` 共 200 行及运行命令] 另用单样本测试了官方 seeded random FPS 起点，避免把受控 parity 协议伪装为官方随机协议。[verified: `artifacts/p3sam_trueofficialfps_cuda_mlx_comparison.json`]

X-Part 论文使用 CD、F-Score@0.1 与 F-Score@0.05，对象归一化到 `[-1,1]`，测试 0/90/180/270 度并取最佳朝向；论文 part-decomposition 为 0.11/0.80/0.71。[verified: `.upstream/papers/x-part.txt:357-394`] 公开仓库声明发布的是 X-Part light version，不是论文 full version。[verified: `.upstream/hunyuan3d-part/README.md:39-40`]

## 5. P3-SAM 200 样本 MLX 结果

| 类别 | 样本数 | mIoU | 平均秒/样本 | 证据 |
|---|---:|---:|---:|---|
| Human-Shape | 29 | 41.0833 | 35.3437 | [verified: `paper_summary.json` → `categories.Human-Shape`] |
| Animals | 23 | 33.6576 | 35.5326 | [verified: `paper_summary.json` → `categories.Animals`] |
| Daily-Used | 25 | 47.5863 | 35.3646 | [verified: `paper_summary.json` → `categories.Daily-Used`] |
| Buildings&&Outdoor | 25 | 32.2072 | 36.4125 | [verified: `paper_summary.json` → `categories.Buildings&&Outdoor`] |
| Transportations | 38 | 35.1907 | 35.7170 | [verified: `paper_summary.json` → `categories.Transportations`] |
| Plants | 18 | 43.8456 | 35.6415 | [verified: `paper_summary.json` → `categories.Plants`] |
| Food | 8 | 43.8215 | 35.7388 | [verified: `paper_summary.json` → `categories.Food`] |
| Electronics | 34 | 46.0317 | 36.0005 | [verified: `paper_summary.json` → `categories.Electronics`] |
| **论文类别宏平均** | **200** | **40.4280** | **35.7268** | [verified: `paper_summary.json` → `paper_macro_instance_miou`] |

shape-micro mIoU 为 40.0125%，平均峰值 MLX 内存为 21.976 GB。[verified: `artifacts/partobj_full_mlx_corrected/paper_summary.json`] 与论文无 connectivity 59.88 相比，类别宏平均低 19.452 个百分点。[verified: 同一 JSON 与 `.upstream/papers/p3-sam.txt:324`]

## 6. CUDA 与 MLX 差异

### 相同 7 个 UID 的受控对照

CUDA 参考实现受数据相关长尾影响：`02e777bd…` 超过 120 秒、`01fe14f5…` 超过 180 秒仍未产出，因此没有把不完整 CUDA 结果包装成 200 样本统计。[verified: 两次独立 CUDA 运行终止后 `records.jsonl` 未增加] 下表只比较 7 个覆盖 7 类且两端均完成的相同 UID；Plants 不在该集合。[verified: `artifacts/partobj_corrected_cuda_stratified/records.jsonl`]

| 指标 | CUDA | MLX | 差异 | 证据 |
|---|---:|---:|---:|---|
| 平均 mIoU | 33.3918 | 33.5199 | MLX +0.1282 点 | [verified: `p3sam_corrected_cuda_mlx_selected7_comparison.json`] |
| 平均推理耗时 | 27.5776 s | 35.5022 s | MLX 1.287× | [verified: 同一 JSON] |
| 平均峰值内存 | 8.1520 GB | 21.9758 GB | MLX 2.695× | [verified: 同一 JSON] |
| 平均 ARI | — | — | 0.7627 | [verified: 同一 JSON] |
| 平均 NMI | — | — | 0.8232 | [verified: 同一 JSON] |
| 对称最佳 mask IoU | — | — | 0.7384 | [verified: 同一 JSON] |

7 个样本的 ARI 从 0.4823 到 0.9968，说明有的形状几乎一致，有的形状差异很大。[verified: `p3sam_corrected_cuda_mlx_selected7_comparison.json` → `samples`] 受控对照的均值质量接近不代表逐面标签等价。[opinion]

官方 FPS 起点单样本中，CUDA/MLX mIoU 为 27.6488/35.7263，耗时为 27.9095/36.4227 秒，ARI 为 0.6338。[verified: `artifacts/p3sam_trueofficialfps_cuda_mlx_comparison.json`]

### 数值定位

20K 点 Sonata feature 对照为 max abs 0.6315、mean abs 0.006132、RMSE 0.008706、P99 0.02450。[verified: `artifacts/p3sam_corrected_cuda_mlx_comparison.json` → `sonata_feature_comparison_20000_points`] fused/显式 attention、einsum/顺序 SubMConv、手写/fused LayerNorm 三组 A/B 均未缩小误差。[verified: `artifacts/p3sam_features_corrected_mlx_*.npy` 对照记录] P3-SAM 的硬阈值与 NMS 会把 feature 数值漂移放大成不同 partition。[opinion]

| X-Part 子模块 | 最大绝对误差 | 平均绝对误差 | RMSE | 证据 |
|---|---:|---:|---:|---|
| PointCross encoder | 0.0625 | 0.006472 | 0.008611 | [verified: `artifacts/xpart_encoder_*_compare`] |
| PartFormer B=1 | 0.008789 | 0.002236 | 0.002796 | [verified: `artifacts/xpart_partformer_*_compare`] |
| PartFormer B=4 | 0.009766 | 0.002136 | 0.002691 | [verified: `artifacts/xpart_partformer_*_compare_p4`] |
| ShapeVAE latent feature | 3.0 | 0.016821 | 0.026067 | [verified: `artifacts/xpart_shapevae_*_compare_real`] |
| ShapeVAE SDF | 0.049805 | 0.000227 | 0.002039 | [verified: `artifacts/xpart_shapevae_*_compare_real`] |

ShapeVAE SDF 符号一致率为 99.9756%，但中间 BF16 feature 不是逐 bit 一致。[verified: `artifacts/xpart_shapevae_*_compare_real` 数值对照]

## 7. X-Part 真实实例与论文表面指标

输入为 `00200996b8f34f55a2dd2f44d316d107.glb`；配置为 P3-SAM 100K/400、82K object/part surface points、4096 condition tokens、50 flow steps、10 parts、seed 42。[verified: `artifacts/xpart_corrected_steps50_r128/runtime.json` 与运行命令]

| 项目 | 结果 | 证据 |
|---|---:|---|
| P3-SAM 与 part 采样 | 37.5982 s | [verified: `runtime.json` → `stage_seconds.predict_and_sample_parts`] |
| Conditioner | 12.4111 s | [verified: `runtime.json` → `stage_seconds.encode_condition`] |
| 50 步扩散 | 174.1365 s | [verified: `runtime.json` → `stage_seconds.diffusion`] |
| resolution 128 解码 | 4.8277 s | [verified: `runtime.json` → `stage_seconds.decode_parts`] |
| 全链路 | 230.3896 s | [verified: `runtime.json` → `total_seconds`] |
| 全链路峰值内存 | 29.4949 GB | [verified: `runtime.json` → `peak_memory_bytes`] |
| resolution 512 重解码 | 48.2589 s | [verified: `artifacts/xpart_corrected_steps50_r512/decode_runtime.json`] |

| 分辨率 | 顶点 | 面 | CD↓ | F@0.1↑ | F@0.05↑ | 全部 watertight | 证据 |
|---|---:|---:|---:|---:|---:|---|---|
| 128 | 73,754 | 144,477 | 0.011736 | 0.997659 | 0.946388 | 否 | [verified: `artifacts/xpart_corrected_steps50_r128/surface_metrics.json`] |
| 512 | 1,375,230 | 2,624,969 | 0.009632 | 0.997829 | 0.991922 | 否 | [verified: `artifacts/xpart_corrected_steps50_r512/surface_metrics.json`] |

这里的 CD/F-Score 按论文的归一化与四方向规则计算，但对象是“生成 parts 合并表面”对“输入 mesh”的单样本整体 sanity check。[verified: `scripts/evaluate_xpart_surface.py` → `best_rotated_surface_metrics`] 它不是论文 200 样本、GT part geometry correspondence 的 part-decomposition 指标，不能与论文 0.11/0.80/0.71 直接比较。[opinion]

resolution 512 明显减少了 resolution 128 的孔洞与薄片，但预览仍可见 part 交叠、局部缺口和非 watertight 几何。[verified: `artifacts/xpart_corrected_steps50_r512/preview/view_00.png` 与 `surface_metrics.json`] 用于 3D 打印或布尔运算前，应追加 self-intersection、manifold 与 watertight 修复。[opinion]

## 8. Mac 部署与复跑

以下命令已在目标 Mac 的部署目录验证。[verified: 本次 SSH 运行记录]

```bash
ssh mt@10.20.134.22
cd /Users/mt/hamster/3D_Split

uv sync --extra hunyuan-mlx

export HF_ENDPOINT=https://hf-mirror.com
hf download tencent/Hunyuan3D-Part \
  --local-dir /Users/mt/hamster/models/Hunyuan3D-Part

# P3-SAM 真实分割
.venv/bin/python scripts/run_p3sam_mlx.py \
  models/PartObjaverse-Tiny/PartObjaverse-Tiny_mesh/00200996b8f34f55a2dd2f44d316d107.glb \
  --weights models/p3sam.safetensors \
  --output artifacts/p3sam_example \
  --points 100000 --prompts 400 --prompt-batch-size 8 --seed 42

# X-Part 完整 50 步生成
.venv/bin/python scripts/run_xpart_mlx.py \
  models/PartObjaverse-Tiny/PartObjaverse-Tiny_mesh/00200996b8f34f55a2dd2f44d316d107.glb \
  --model-dir /Users/mt/hamster/models/Hunyuan3D-Part \
  --p3-weights models/p3sam.safetensors \
  --output artifacts/xpart_output \
  --points 100000 --prompts 400 --prompt-batch-size 8 \
  --surface-points 81920 --steps 50 --resolution 128 --seed 42

# 对保存的 latents 按官方默认 resolution 512 重解码
.venv/bin/python scripts/decode_xpart_latents_mlx.py \
  models/PartObjaverse-Tiny/PartObjaverse-Tiny_mesh/00200996b8f34f55a2dd2f44d316d107.glb \
  --latents artifacts/xpart_output/latents.npy \
  --weights /Users/mt/hamster/models/Hunyuan3D-Part/shapevae/shapevae.safetensors \
  --output artifacts/xpart_output_r512 --resolution 512
```

如需官方 seeded random FPS 起点，在 P3-SAM 或 X-Part 命令追加 `--official-fps-start`；不追加时固定起点 0，便于跨后端受控比较。[verified: `scripts/run_p3sam_mlx.py`、`scripts/run_xpart_mlx.py` 参数定义]

## 9. 限制与上线建议

1. P3-SAM MLX 全量宏平均为 40.43%，不能宣称达到论文 59.88%。[verified: `paper_summary.json` 与论文表 2]
2. CUDA/MLX 的 7 样本均值 mIoU 接近，但 ARI 只有 0.7627，不能依赖逐面标签完全一致。[verified: `p3sam_corrected_cuda_mlx_selected7_comparison.json`]
3. connectivity 后处理尚未进入 MLX 主路径；当前结果对应论文无 connectivity 任务，并满足 X-Part bbox 输入链路。[verified: `src/split3d/hunyuan/p3sam_pipeline.py` 与运行配置]
4. 公布的是 X-Part light version，论文 full-version 数字只能当参考。[verified: `.upstream/hunyuan3d-part/README.md:39-40`]
5. resolution 512 输出仍非全部 watertight；资产生产前需要几何修复与人工验收。[verified: `xpart_corrected_steps50_r512/surface_metrics.json`]
6. Tencent Hunyuan 3D-Part Community License 排除欧盟、英国和韩国，分发要求 NOTICE，发布时全产品月活超过 100 万需另行申请授权。[verified: `.upstream/hunyuan3d-part/LICENSE:3,17,23-31`] 上线前应让法务按部署地域、分发方式与规模审核。[opinion]

## 10. 证据与最终产物

- P3 全量论文汇总：`artifacts/partobj_full_mlx_corrected/paper_summary.json`。[verified: 文件存在且 `completed=expected=200`]
- P3 全量逐样本记录：`artifacts/partobj_full_mlx_corrected/records.jsonl`。[verified: 文件为 200 行]
- CUDA/MLX 7 样本对照：`artifacts/p3sam_corrected_cuda_mlx_selected7_comparison.json`。[verified: `matched_samples=7`]
- 官方 FPS 单样本对照：`artifacts/p3sam_trueofficialfps_cuda_mlx_comparison.json`。[verified: `matched_samples=1`]
- P3 CUDA/MLX 分割预览：`artifacts/p3sam_corrected_cuda_visual/preview/view_00.png` 与 `artifacts/p3sam_corrected_mlx_visual/preview/view_00.png`。[verified: 两个 PNG 已人工查看]
- X-Part resolution 128 GLB：`artifacts/xpart_corrected_steps50_r128/xpart_scene.glb`。[verified: 文件已从 Mac 下载]
- X-Part resolution 512 GLB：`artifacts/xpart_corrected_steps50_r512/xpart_scene.glb`。[verified: 文件已从 Mac 下载]
- X-Part resolution 512 预览：`artifacts/xpart_corrected_steps50_r512/preview/view_00.png`。[verified: PNG 已人工查看]
- 官方论文：P3-SAM `arXiv:2509.06784`；X-Part `arXiv:2509.08643`。[verified: `.upstream/papers/p3-sam.pdf` 与 `.upstream/papers/x-part.pdf`]

代码质量验证：Ruff 全部通过，Pytest 30 passed、1 skipped；跳过项是条件环境测试。[verified: 最终本地 `ruff check src scripts tests` 与 `pytest -q` 输出]
