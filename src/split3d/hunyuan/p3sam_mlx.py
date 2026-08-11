from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from split3d.hunyuan.sonata_data import prepare_sonata_input
from split3d.hunyuan.sonata_mlx import SonataFeatureExtractorMLX

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - exercised by CUDA/CPU reference hosts
    mx = None


def _require_mlx() -> None:
    if mx is None:
        raise RuntimeError("P3SAMMLX requires Apple MLX")


def _linear(x: Any, weight: Any, bias: Any) -> Any:
    return x @ weight.T + bias


def _gelu(x: Any) -> Any:
    return x * 0.5 * (1.0 + mx.erf(x / np.sqrt(2.0)))


class MLPMLX:
    def __init__(self, weights: dict[str, Any], prefix: str) -> None:
        self.weights = [weights[f"{prefix}.{index}.weight"] for index in (0, 2, 4)]
        self.biases = [weights[f"{prefix}.{index}.bias"] for index in (0, 2, 4)]

    def __call__(self, x: Any) -> Any:
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases, strict=True)):
            x = _linear(x, weight, bias)
            if index < 2:
                x = _gelu(x)
        return x


@dataclass(frozen=True)
class P3SAMMasks:
    masks: tuple[Any, Any, Any]
    predicted_iou: Any


class P3SAMMLX:
    """Native MLX P3-SAM encoder and point-promptable mask decoder."""

    def __init__(self, weights: dict[str, Any]) -> None:
        _require_mlx()
        self.feature_extractor = SonataFeatureExtractorMLX(weights, prefix="")
        self.stage1 = tuple(MLPMLX(weights, f"seg_mlp_{index}") for index in (1, 2, 3))
        self.stage2_global = MLPMLX(weights, "seg_s2_mlp_g")
        self.stage2 = tuple(MLPMLX(weights, f"seg_s2_mlp_{index}") for index in (1, 2, 3))
        self.iou = MLPMLX(weights, "iou_mlp")
        self.iou_out = MLPMLX(weights, "iou_mlp_out")

    @classmethod
    def from_safetensors(cls, path: str | Path) -> P3SAMMLX:
        _require_mlx()
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        return cls(mx.load(str(path)))

    def extract_features(
        self,
        points: np.ndarray,
        normals: np.ndarray | None = None,
        *,
        seed: int = 42,
    ) -> Any:
        prepared = prepare_sonata_input(points, normals, seed=seed)
        return self.feature_extractor(prepared)

    def predict_masks(
        self,
        features: Any,
        points: np.ndarray,
        prompts: np.ndarray,
        *,
        iterations: int = 1,
        prompt_batch_size: int = 1,
    ) -> P3SAMMasks:
        points_array = mx.array(np.asarray(points, dtype=np.float32))
        prompts_array = mx.array(np.asarray(prompts, dtype=np.float32))
        if features.ndim != 2 or features.shape[0] != points_array.shape[0] or features.shape[1] != 512:
            raise ValueError(f"features must have shape [N, 512], got {features.shape}")
        if prompts_array.ndim != 2 or prompts_array.shape[1] != 3:
            raise ValueError(f"prompts must have shape [K, 3], got {prompts_array.shape}")
        if iterations < 1:
            raise ValueError("iterations must be at least one")
        if prompt_batch_size < 1:
            raise ValueError("prompt_batch_size must be positive")
        if prompt_batch_size > 8:
            raise ValueError(
                "prompt_batch_size above 8 is disabled because MLX 0.32.0 produced batch-dependent masks"
            )

        mask_chunks: list[list[np.ndarray]] = [[], [], []]
        iou_chunks: list[np.ndarray] = []
        point_count = points_array.shape[0]
        for start in range(0, prompts_array.shape[0], prompt_batch_size):
            prompt_chunk = prompts_array[start : start + prompt_batch_size]
            prompt_count = prompt_chunk.shape[0]
            feature_grid = mx.broadcast_to(features[:, None, :], (point_count, prompt_count, 512))
            point_grid = mx.broadcast_to(points_array[:, None, :], (point_count, prompt_count, 3))
            prompt_grid = mx.broadcast_to(prompt_chunk[None, :, :], (point_count, prompt_count, 3))
            base = mx.concatenate([feature_grid, point_grid, prompt_grid], axis=-1)
            logits = mx.stack([head(base)[..., 0] for head in self.stage1], axis=-1)
            global_features = None
            stage2_input = None
            for _ in range(iterations):
                stage2_base = mx.concatenate([base, logits], axis=-1)
                global_features = mx.max(self.stage2_global(stage2_base), axis=0)
                global_grid = mx.broadcast_to(global_features[None, :, :], (point_count, prompt_count, 256))
                stage2_input = mx.concatenate([global_grid, stage2_base], axis=-1)
                logits = mx.stack([head(stage2_input)[..., 0] for head in self.stage2], axis=-1)
            if global_features is None or stage2_input is None:
                raise RuntimeError("stage-two decoder did not run")
            probabilities = mx.sigmoid(logits)
            iou_features = mx.max(self.iou(stage2_input), axis=0)
            predicted_iou = mx.sigmoid(self.iou_out(iou_features))
            mx.eval(probabilities, predicted_iou)
            probabilities_cpu = np.array(probabilities)
            for index in range(3):
                mask_chunks[index].append(probabilities_cpu[..., index].T)
            iou_chunks.append(np.array(predicted_iou))
            del (
                base,
                feature_grid,
                global_features,
                global_grid,
                iou_features,
                logits,
                point_grid,
                probabilities,
                predicted_iou,
                prompt_grid,
                stage2_base,
                stage2_input,
            )
        return P3SAMMasks(
            masks=tuple(np.concatenate(chunks, axis=0) for chunks in mask_chunks),
            predicted_iou=np.concatenate(iou_chunks, axis=0),
        )
