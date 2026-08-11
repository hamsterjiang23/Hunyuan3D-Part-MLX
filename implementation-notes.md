# Implementation notes

## Deviations

- 计划阶段错误假定可用 48–80GB 显存；实测机器为 RTX 4070 12GB，因此视觉模型层保持可替换，自动选择 CPU/CUDA。
- PyPI 环境解析出了 CPU 版 Torch 2.13；项目配置后来改为显式锁定官方 cu128 索引的 Torch 2.8，SAM2 手工预处理后不再依赖 Torchvision。
- cu128 wheel 已进入本地 uv 临时缓存，但两次同步分别在 10 分钟和 5 分钟边界无日志超时；当前 `.venv` 仍使用 CPU Torch，先完成 CPU 端到端验证。

## Discovered edge cases

- 可复用的 Meshy 样本是单节点、单材质、单几何连通网格；connected-components 只能得到一个整体，不能代替语义拆分。
- 样本的 UV/法线 seam 复制了顶点，直接按属性顶点索引会误判为 73 个连通块；分割邻接临时按空间位置焊接，导出仍保留原属性顶点。
- Transformers 默认选择 `Sam2ImageProcessorFast`，即使 CPU 推理也要求 Torchvision；`use_fast=False` 在当前版本中无效。
- 当前 Transformers 版本只提供 SAM2 fast image processor，无法通过 `use_fast=False` 降级；最终改为在适配器内完成等价的 resize、ImageNet normalize、box scale 和 mask resize，移除运行时 Torchvision 依赖。
- 直接逐面投票会让全物体 `body` mask 压过只在少数视角出现的局部部件；分数现按各语义实际被检测到的视角归一化，并对小 mask 加有限的 specificity 权重。
- 自动标签会产生 1–5 面的碎岛；导出前把小于总面数 0.5%（至少 8 面）的区域并入共享边界最多的相邻标签。
- 用户目标是每个语义一个游戏资产，不是每个几何连通岛一个资产；同一标签现在只导出一个节点。

## Questions for review

- 使用用户实际的马模型验证 `head,body,legs,tail` 四类边界；当前动物代理样本是低模兔子。

## Session summary

- Deviations recorded: 3.
- Most likely revisit: automatic semantic boundary quality on the user's real horse asset.
- Edge cases found: 7 recorded cases covering fused meshes, UV seams, SAM2 preprocessing, score bias, face islands, and semantic grouping.
- First file for the next session: `artifacts/bunny-semantic-split/preview.png`, then `src/split3d/vision.py`.
- Current handoff: runnable CPU end-to-end prototype with CUDA dependency locked but not installed in `.venv`.
