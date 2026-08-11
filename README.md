# Hunyuan3D-Part-MLX

Hunyuan3D-Part 的原生 Apple MLX 移植，覆盖两条真实网格推理链路：

- P3-SAM：从 GLB/OBJ 网格预测面实例分割、包围盒和带颜色的分割 GLB。
- X-Part：P3-SAM → Conditioner → PartFormer → ShapeVAE，生成独立 part geometry 并导出 GLB。

MLX 神经网络路径不依赖 PyTorch/CUDA；P3-SAM 完整路径使用官方源面投票、邻接 flood-fill、连通域过滤、缺失部件补齐和累计面积后处理，Marching Cubes 与 GLB I/O 使用 NumPy、scikit-image 和 trimesh。

## 当前状态

完整公开权重共 9,492,999,106 字节（约 9.49 GB / 8.84 GiB）：P3-SAM 451 MB、PartFormer 6.63 GB、Conditioner 1.76 GB、ShapeVAE 656 MB。

旧版无 connectivity 基线在 PartObjaverse-Tiny 200 样本上的 P3-SAM MLX 类别宏平均 mIoU 为 40.43%；该数字不能代表当前完整官方后处理路径。当前实现已补齐官方 topology/postprocess 语义，新的 200 样本论文协议结果必须在 Mac 上重新评测后才能声明。

官方 Sonata FlashAttention 会将 QKV 转为 FP16；但 5 个相同 100K 样本的 CUDA A/B 中，该路径比 FP32 平均低 1.07 mIoU，MLX 0.32.0 直接使用 FP16 SDPA 还会产生全 NaN feature，因此默认保留实测更准、更稳定的 FP32 attention。完整误差、耗时和内存数据见 [`reports/p3sam-attention-precision-ab.json`](reports/p3sam-attention-precision-ab.json)。官方仓库的 [PartObjaverse-Tiny 复现 issue #13](https://github.com/Tencent-Hunyuan/Hunyuan3D-Part/issues/13) 也报告公开 `auto_mask.py` 约 60% 而非论文 81.14%，目前没有官方回复；本项目不会把公开 demo 结果误标成论文复现。

论文的 81.14 connectivity 指标使用了每个 GT part 的随机 prompt，属于带真实部件提示的强条件，并非公开自动分割 demo。项目现提供独立的论文描述协议审计器；固定单样本中，GT-prompt 路径把 MLX FP32 最终 mIoU 从 37.54 提升到 48.12，但仍未复现 81.14。一个不依赖 GT 的质心 prompt 二次细化实验反而降至 31.48，已从推理主路径移除。完整单样本数据和 147 样本停止点见 [`reports/p3sam-paper-protocol-single-sample.json`](reports/p3sam-paper-protocol-single-sample.json)；147 条是用户终止前的非随机前缀，只可作为诊断，不可作为论文全量指标。

完整方法、硬件、CUDA/MLX 数值差异和逐模块误差见 [移植与评测报告](reports/Hunyuan3D-Part-MLX-port-report.md)。

## 安装

要求 Apple Silicon Mac、Python 3.11/3.12 和足够的统一内存。

```bash
uv sync --extra hunyuan-mlx --extra service --extra dev

export HF_ENDPOINT=https://hf-mirror.com
hf download tencent/Hunyuan3D-Part \
  --local-dir /Users/mt/hamster/models/Hunyuan3D-Part
```

P3-SAM 权重路径可单独指定；当前部署使用 `models/p3sam.safetensors`。

## 命令行推理

```bash
# P3-SAM 面实例分割
.venv/bin/python scripts/run_p3sam_mlx.py input.glb \
  --weights models/p3sam.safetensors \
  --output artifacts/p3sam_output \
  --points 100000 --prompts 400 --prompt-batch-size 1 --seed 42

# X-Part 完整生成
.venv/bin/python scripts/run_xpart_mlx.py input.glb \
  --model-dir /Users/mt/hamster/models/Hunyuan3D-Part \
  --p3-weights models/p3sam.safetensors \
  --output artifacts/xpart_output \
  --points 100000 --prompts 400 --prompt-batch-size 1 \
  --surface-points 81920 --steps 50 --resolution 128 --seed 42
```

P3-SAM 默认启用官方 seeded FPS、mesh cleaning、connectivity 和 `threshold=0.95` 的完整后处理。每次输出包含 `segmented_projected.glb`、`segmented_connectivity.glb`、最终 `segmented.glb`、三阶段 `face_ids*.npy`、AABB 场景以及逐 part GLB。

MLX 默认逐 prompt 解码（`prompt_batch_size=1`）。固定 replay 验证 batch 1 与 batch 8 的二值 mask 和最终逐面标签完全一致，同时内存更低；MLX 0.32.0 的 batch 32 会产生显著不同的 mask，因此公共接口限制为 1–8。

需要 CUDA/MLX 逐阶段对拍时，先用一端生成 `--trace-dir TRACE`，另一端传入 `--replay-manifest TRACE/replay_manifest.npz`；再运行 `scripts/compare_p3sam_traces.py`。追加 `--trace-full-tensors` 会同时保存 Sonata 特征和三头 mask 概率，文件体积较大。

## 本地 MLX worker

worker 默认只监听 `127.0.0.1:8083`，输入路径必须位于统一队列的 input root 内；模型按需加载，并提供显存/统一内存卸载接口。

```bash
.venv/bin/python scripts/serve_hunyuan_part_mlx.py \
  --host 127.0.0.1 --port 8083 \
  --cache-path server_cache \
  --input-root /Users/mt/hamster/3d-inference/server_cache/inputs \
  --model-dir /Users/mt/hamster/models/Hunyuan3D-Part \
  --p3-weights models/p3sam.safetensors
```

内部接口：

- `POST /send`：提交共享目录中的网格路径和推理参数。
- `GET /status/{uid}`：查询状态与当前阶段。
- `GET /download/{uid}`：下载主 GLB。
- `GET /bundle/{uid}`：下载全部产物 ZIP。
- `POST /unload`：空闲时释放 MLX 模型与缓存。
- `GET /health`：查询 worker、忙碌状态和模型加载状态。

对外部署时应通过现有 `8080` 持久化队列访问，不要公开 `8083`。队列使用二进制请求体，避免将大网格转换为 Base64：

仓库中的 `deploy/3d-inference/` 保存了本次 Mac 部署使用的队列入口、SQLite 存储层、五服务启动器、API 文档和 Part 协议测试，可覆盖到现有 `3d-inference` 服务目录后使用；启动器中的用户目录和局域网地址需要按目标机器调整。

```bash
curl -X POST \
  'http://MAC_IP:8080/submit/part?filename=input.glb&mode=segment' \
  -H 'Content-Type: model/gltf-binary' \
  -H "X-API-Key: $INFERENCE_API_KEY" \
  --data-binary @input.glb

curl -H "X-API-Key: $INFERENCE_API_KEY" \
  http://MAC_IP:8080/status/TASK_UID

curl -L -H "X-API-Key: $INFERENCE_API_KEY" \
  http://MAC_IP:8080/download/TASK_UID -o segmented.glb

curl -L -H "X-API-Key: $INFERENCE_API_KEY" \
  http://MAC_IP:8080/bundle/TASK_UID -o artifacts.zip
```

`mode=segment` 只加载 P3-SAM；`mode=generate_parts` 会加载完整 X-Part，并可用 query 参数覆盖 `steps`、`resolution`、`surface_points` 等默认值。结果由队列持久化并按部署侧保留策略清理。

## 测试

```bash
uv run ruff check src scripts tests
uv run pytest -q
```

## 已知限制

- 40.43% 是旧版无 connectivity 基线；完整官方后处理版本的 200 样本指标正在重新生成，完成前不能宣称已经达到论文 81.14%。
- 官方公开的是 X-Part light version，论文 full-version 数字只能作为参考。
- MLX 与 CUDA 的浮点和稀疏算子差异会被 mask 阈值与 NMS 放大，均值接近不代表实例标签等价。
- X-Part 导出的 part 并不保证全部 watertight。
