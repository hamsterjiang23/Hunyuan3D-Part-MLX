from __future__ import annotations

import argparse
import json
import sys
import time
import types
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import trimesh
from safetensors.torch import load_file

from split3d.hunyuan.p3sam_mlx import P3SAMMasks
from split3d.hunyuan.p3sam_pipeline import save_segmentation, segment_mesh
from split3d.hunyuan.sonata_data import prepare_sonata_input


def _install_torch_scatter_fallback() -> None:
    module = types.ModuleType("torch_scatter")

    def segment_csr(source: torch.Tensor, indptr: torch.Tensor, reduce: str = "sum") -> torch.Tensor:
        lengths = indptr[1:] - indptr[:-1]
        return torch.segment_reduce(source, reduce, lengths=lengths)

    module.segment_csr = segment_csr  # type: ignore[attr-defined]
    sys.modules["torch_scatter"] = module


def _install_flash_attn_fallback() -> None:
    """Expose the released FP16 attention contract through PyTorch SDPA.

    The official Sonata path always casts packed QKV to FP16 before calling
    ``flash_attn_varlen_qkvpacked_func`` and casts its result back to the
    surrounding FP32 model.  FlashAttention is not available on the Windows
    CUDA reference host, but PyTorch SDPA provides the same dtype boundary and
    selects an accelerated attention kernel when the device supports it.
    """

    module = types.ModuleType("flash_attn")

    def flash_attn_varlen_qkvpacked_func(
        qkv: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        dropout_p: float = 0.0,
        softmax_scale: float | None = None,
        **_: object,
    ) -> torch.Tensor:
        boundaries = cu_seqlens.detach().cpu().tolist()
        lengths = np.diff(boundaries)
        chunks: list[torch.Tensor | None] = [None] * len(lengths)
        for length in np.unique(lengths):
            sequence_indices = np.flatnonzero(lengths == length)
            packed = torch.stack(
                [qkv[boundaries[index] : boundaries[index + 1]] for index in sequence_indices],
                dim=0,
            )
            query = packed[:, :, 0].permute(0, 2, 1, 3)
            key = packed[:, :, 1].permute(0, 2, 1, 3)
            value = packed[:, :, 2].permute(0, 2, 1, 3)
            attended = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=dropout_p,
                scale=softmax_scale,
            )
            attended = attended.permute(0, 2, 1, 3)
            for output_index, sequence_index in enumerate(sequence_indices):
                chunks[int(sequence_index)] = attended[output_index]
        if any(chunk is None for chunk in chunks):
            raise RuntimeError("variable-length attention omitted an input sequence")
        result = torch.cat([chunk for chunk in chunks if chunk is not None], dim=0)
        if result.shape[0] != qkv.shape[0] or max_seqlen <= 0:
            raise RuntimeError("invalid variable-length attention output")
        return result

    module.flash_attn_varlen_qkvpacked_func = flash_attn_varlen_qkvpacked_func  # type: ignore[attr-defined]
    sys.modules["flash_attn"] = module


def _mlp(*channels: int) -> nn.Sequential:
    modules: list[nn.Module] = []
    for index, (input_channels, output_channels) in enumerate(zip(channels[:-1], channels[1:], strict=True)):
        modules.append(nn.Linear(input_channels, output_channels))
        if index < len(channels) - 2:
            modules.append(nn.GELU())
    return nn.Sequential(*modules)


class P3SAMCUDA(nn.Module):
    def __init__(self, upstream_root: Path, *, official_attention_precision: bool = False) -> None:
        super().__init__()
        _install_torch_scatter_fallback()
        if official_attention_precision:
            _install_flash_attn_fallback()
        partgen = upstream_root / "XPart" / "partgen"
        sys.path.insert(0, str(partgen))
        from models.sonata.model import PointTransformerV3

        config = json.loads((partgen / "config" / "sonata.json").read_text(encoding="utf-8"))
        config["enable_flash"] = official_attention_precision
        config["shuffle_orders"] = True
        self.sonata = PointTransformerV3(**config)
        import spconv.pytorch as spconv
        from spconv.core import ConvAlgo

        for module in self.sonata.modules():
            if isinstance(module, spconv.SubMConv3d):
                module.algo = ConvAlgo.Native
        self.mlp = _mlp(1232, 512, 512, 512)
        self.seg_mlp_1 = _mlp(518, 512, 512, 1)
        self.seg_mlp_2 = _mlp(518, 512, 512, 1)
        self.seg_mlp_3 = _mlp(518, 512, 512, 1)
        self.seg_s2_mlp_g = _mlp(521, 256, 256, 256)
        self.seg_s2_mlp_1 = _mlp(777, 256, 256, 1)
        self.seg_s2_mlp_2 = _mlp(777, 256, 256, 1)
        self.seg_s2_mlp_3 = _mlp(777, 256, 256, 1)
        self.iou_mlp = _mlp(777, 256, 256, 256)
        self.iou_mlp_out = _mlp(256, 256, 256, 3)

    def extract_features(
        self, points: np.ndarray, normals: np.ndarray | None = None, *, seed: int = 42
    ) -> torch.Tensor:
        prepared = prepare_sonata_input(points, normals, seed=seed)
        torch.manual_seed(seed)
        device = next(self.parameters()).device
        data = {
            "coord": torch.from_numpy(prepared.coord).to(device),
            "grid_coord": torch.from_numpy(prepared.grid_coord).to(device),
            "color": torch.from_numpy(prepared.color).to(device),
            "inverse": torch.from_numpy(prepared.inverse).to(device),
            "feat": torch.from_numpy(prepared.feat).to(device),
            "batch": torch.from_numpy(prepared.batch).to(device),
            "offset": torch.from_numpy(prepared.offset).to(device),
        }
        point = self.sonata(data)
        while "pooling_parent" in point:
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        return self.mlp(point.feat)[point.inverse]

    def predict_masks(
        self,
        features: torch.Tensor,
        points: np.ndarray,
        prompts: np.ndarray,
        *,
        iterations: int = 1,
        prompt_batch_size: int = 32,
    ) -> P3SAMMasks:
        device = features.device
        point_tensor = torch.from_numpy(np.asarray(points, dtype=np.float32)).to(device)
        prompt_tensor = torch.from_numpy(np.asarray(prompts, dtype=np.float32)).to(device)
        mask_chunks: list[list[torch.Tensor]] = [[], [], []]
        iou_chunks = []
        for start in range(0, len(prompt_tensor), prompt_batch_size):
            prompt_chunk = prompt_tensor[start : start + prompt_batch_size]
            point_count, prompt_count = len(point_tensor), len(prompt_chunk)
            base = torch.cat(
                [
                    features[:, None, :].expand(point_count, prompt_count, 512),
                    point_tensor[:, None, :].expand(point_count, prompt_count, 3),
                    prompt_chunk[None, :, :].expand(point_count, prompt_count, 3),
                ],
                dim=-1,
            )
            logits = torch.stack(
                [self.seg_mlp_1(base)[..., 0], self.seg_mlp_2(base)[..., 0], self.seg_mlp_3(base)[..., 0]],
                dim=-1,
            )
            stage2_input = None
            for _ in range(iterations):
                stage2_base = torch.cat([base, logits], dim=-1)
                global_features = self.seg_s2_mlp_g(stage2_base).max(dim=0).values
                stage2_input = torch.cat(
                    [global_features[None, :, :].expand(point_count, prompt_count, 256), stage2_base],
                    dim=-1,
                )
                logits = torch.stack(
                    [
                        self.seg_s2_mlp_1(stage2_input)[..., 0],
                        self.seg_s2_mlp_2(stage2_input)[..., 0],
                        self.seg_s2_mlp_3(stage2_input)[..., 0],
                    ],
                    dim=-1,
                )
            if stage2_input is None:
                raise RuntimeError("stage-two decoder did not run")
            probabilities = torch.sigmoid(logits)
            for index in range(3):
                mask_chunks[index].append(probabilities[..., index].T.cpu())
            iou_features = self.iou_mlp(stage2_input).max(dim=0).values
            iou_chunks.append(torch.sigmoid(self.iou_mlp_out(iou_features)).cpu())
            del base, global_features, iou_features, logits, probabilities, stage2_base, stage2_input
        return P3SAMMasks(
            masks=tuple(torch.cat(chunks, dim=0).numpy() for chunks in mask_chunks),
            predicted_iou=torch.cat(iou_chunks, dim=0).numpy(),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synchronized CUDA P3-SAM reference inference")
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=Path(".upstream/hunyuan3d-part"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points", type=int, default=100_000)
    parser.add_argument("--prompts", type=int, default=400)
    parser.add_argument("--prompt-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--official-fps-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean-mesh", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--connectivity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--postprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--postprocess-threshold", type=float, default=0.95)
    parser.add_argument("--replay-manifest", type=Path)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--trace-full-tensors", action="store_true")
    parser.add_argument(
        "--official-attention-precision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Match the release's FP16 QKV attention boundary",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = P3SAMCUDA(args.upstream, official_attention_precision=args.official_attention_precision)
    model.load_state_dict(load_file(args.weights), strict=True)
    model.cuda().eval()
    load_seconds = time.perf_counter() - started
    mesh = trimesh.load(args.mesh, force="mesh")
    inference_started = time.perf_counter()
    with torch.inference_mode():
        result = segment_mesh(
            model,
            mesh,
            point_count=args.points,
            prompt_count=args.prompts,
            prompt_batch_size=args.prompt_batch_size,
            seed=args.seed,
            prompt_start_index=None if args.official_fps_start else 0,
            clean_mesh=args.clean_mesh,
            connectivity=args.connectivity,
            postprocess=args.postprocess,
            postprocess_threshold=args.postprocess_threshold,
            replay_manifest=args.replay_manifest,
            trace_dir=args.trace_dir,
            trace_full_tensors=args.trace_full_tensors,
            progress=lambda stage, seconds: print(f"{stage}: {seconds:.3f}s", flush=True),
        )
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    save_segmentation(mesh, result, args.output, seed=args.seed)
    runtime = {
        "backend": "cuda",
        "device": torch.cuda.get_device_name(),
        "weights": str(args.weights),
        "mesh": str(args.mesh),
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "total_seconds": time.perf_counter() - started,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "stage_seconds": result.stage_seconds,
        "diagnostics": asdict(result.diagnostics),
    }
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    print(json.dumps(runtime, indent=2))


if __name__ == "__main__":
    main()
