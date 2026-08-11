# split3d

`split3d` 面向游戏开发，把 AI 生成的单体网格按少量语义名称拆成独立资产，例如把马拆成 `head`、`body`、`legs`、`tail`。它不生成 LOD、碰撞体、打印结构或隐藏内部几何。

当前实现包含：

- GLB、GLTF、OBJ 读取与网格预检；
- 固定多视角 RGB 与逐像素三角面 ID 渲染；
- Grounding DINO 文字部件检测；
- SAM2 检测框分割；
- 2D mask 回投到源三角面、邻接补全和小区域清理；
- 每个语义按相对规模保留主要连通区域，小于该语义 10% 的碎片自动并回相邻部件；
- 每个语义名称导出一个节点的 `split.glb`，并可同时导出各部件 GLB；
- `parts.json`、`face_labels.npy`、`face_scores.npy`、检测记录，以及每个子部件各自的原贴图颜色渲染图。

## 快速开始

```powershell
uv sync --extra dev --extra vision
uv run split3d inspect model.glb
uv run split3d render model.glb --output views --views 12 --resolution 512
uv run split3d auto horse.glb --parts "head,body,legs,tail" --output output
```

默认从 Hugging Face 本地缓存读取 `IDEA-Research/grounding-dino-base`，并从 `models/sam2-hiera-tiny/` 读取 SAM2 tiny。只有显式添加 `--allow-download` 才允许模型库联网。

自动结果是可编辑的粗分资产，不保证直接达到最终美术质量。先查看 `renders/` 中每个部件独立的真实颜色图片；如需手工修正，可编辑 `face_labels.npy` 后重新导出：

```powershell
uv run split3d split horse.glb --face-labels labels.npy --parts "head,body,legs,tail" --output output
```

`labels.npy` 是长度等于三角面数量的一维整数数组。`0,1,2,3` 分别对应 `--parts` 中的名称，`-1` 表示未标注并由网格邻接传播补齐。同一标签即使包含多个不相连区域，也会合并为一个游戏资产节点。

当前不直接读取 FBX；先在 Blender 中转换为 GLB。导出的切口保持开放，不做封口。
