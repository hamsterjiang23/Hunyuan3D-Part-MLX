from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Detection:
    semantic_index: int
    semantic_name: str
    score: float
    box: list[float]


def _device_name(requested: str, torch_module: Any) -> str:
    if requested == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested but the installed Torch build cannot access CUDA")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return requested


class GroundingDinoDetector:
    def __init__(
        self,
        model: str | Path,
        *,
        device: str = "auto",
        local_files_only: bool = True,
        cache_dir: Path | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = _device_name(device, torch)
        model_ref = str(model)
        self.processor = AutoProcessor.from_pretrained(
            model_ref,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_ref,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        ).to(self.device)
        self.model.eval()

    def detect(
        self,
        image: Image.Image,
        part_names: list[str],
        *,
        box_threshold: float,
        text_threshold: float,
    ) -> list[Detection]:
        inputs = self.processor(images=image, text=[part_names], return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        processed = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        result_labels = processed.get("text_labels", processed.get("labels", []))
        detections: list[Detection] = []
        for box, score, result_label in zip(
            processed["boxes"],
            processed["scores"],
            result_labels,
            strict=True,
        ):
            label_text = str(result_label).strip().lower()
            semantic_index = next(
                (
                    index
                    for index, name in enumerate(part_names)
                    if name.lower() == label_text or name.lower() in label_text or label_text in name.lower()
                ),
                -1,
            )
            if semantic_index < 0:
                continue
            detections.append(
                Detection(
                    semantic_index=semantic_index,
                    semantic_name=part_names[semantic_index],
                    score=float(score.item()),
                    box=[float(value) for value in box.tolist()],
                )
            )
        return detections


class Sam2BoxSegmenter:
    def __init__(
        self,
        model: str | Path,
        *,
        device: str = "auto",
        local_files_only: bool = True,
        cache_dir: Path | None = None,
    ) -> None:
        import torch
        from transformers import Sam2Model

        self.torch = torch
        self.device = _device_name(device, torch)
        model_ref = str(model)
        loaded_model: Any = Sam2Model.from_pretrained(
            model_ref,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
        self.model = loaded_model.to(self.device)
        self.model.eval()
        self.target_size = int(self.model.config.image_size)

    def segment(self, image: Image.Image, detections: list[Detection]) -> np.ndarray:
        if not detections:
            return np.zeros((0, image.height, image.width), dtype=bool)
        rgb = image.convert("RGB").resize(
            (self.target_size, self.target_size),
            Image.Resampling.BILINEAR,
        )
        pixels = np.asarray(rgb, dtype=np.float32) / 255.0
        pixel_values = self.torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
        mean = self.torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = self.torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        pixel_values = ((pixel_values - mean) / std).to(self.device)

        input_boxes = self.torch.tensor(
            [[detection.box for detection in detections]],
            dtype=self.torch.float32,
        )
        input_boxes[..., (0, 2)] *= self.target_size / image.width
        input_boxes[..., (1, 3)] *= self.target_size / image.height
        input_boxes = input_boxes.to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(
                pixel_values=pixel_values,
                input_boxes=input_boxes,
                multimask_output=False,
            )
        masks = outputs.pred_masks[0].float()
        masks = self.torch.nn.functional.interpolate(
            masks,
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )
        values = (masks > 0.0).detach().cpu().numpy()
        if values.ndim == 4 and values.shape[1] == 1:
            values = values[:, 0]
        if values.ndim != 3:
            raise RuntimeError(f"unexpected SAM2 mask shape: {values.shape}")
        return values.astype(bool)


def accumulate_face_scores(
    face_ids: np.ndarray,
    detections: list[Detection],
    masks: np.ndarray,
    numerator: np.ndarray,
    visibility: np.ndarray,
    semantic_visibility: np.ndarray | None = None,
) -> None:
    if face_ids.ndim != 2:
        raise ValueError("face_ids must be a 2D array")
    visible_ids = face_ids[face_ids >= 0]
    if visible_ids.size:
        visible_counts = np.bincount(visible_ids, minlength=len(visibility))[: len(visibility)]
        visibility += visible_counts
        if semantic_visibility is not None:
            for semantic_index in {detection.semantic_index for detection in detections}:
                semantic_visibility[:, semantic_index] += visible_counts
    if len(detections) != len(masks):
        raise ValueError("detections and masks must have equal length")
    for detection, mask in zip(detections, masks, strict=True):
        if mask.shape != face_ids.shape:
            raise ValueError("mask and face-id image shapes must match")
        selected = face_ids[mask & (face_ids >= 0)]
        if selected.size == 0:
            continue
        counts = np.bincount(selected, minlength=len(visibility))[: len(visibility)]
        specificity = 1.0
        if semantic_visibility is not None:
            mask_fraction = float(np.count_nonzero(mask)) / mask.size
            specificity = min(3.0, 1.0 / np.sqrt(max(mask_fraction, 0.05)))
        numerator[:, detection.semantic_index] += counts * detection.score * specificity


def infer_face_labels(
    views_dir: Path,
    face_count: int,
    part_names: list[str],
    detector: GroundingDinoDetector,
    segmenter: Sam2BoxSegmenter,
    *,
    box_threshold: float = 0.25,
    text_threshold: float = 0.2,
    min_face_score: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    manifest = json.loads((views_dir / "views.json").read_text(encoding="utf-8"))
    numerator = np.zeros((face_count, len(part_names)), dtype=np.float64)
    visibility = np.zeros(face_count, dtype=np.float64)
    semantic_visibility = np.zeros_like(numerator)
    all_detections: list[dict[str, Any]] = []
    for view in manifest["views"]:
        image = Image.open(views_dir / view["rgb"]).convert("RGB")
        face_ids = np.load(views_dir / view["face_ids"], allow_pickle=False)
        detections = detector.detect(
            image,
            part_names,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        masks = segmenter.segment(image, detections)
        accumulate_face_scores(
            face_ids,
            detections,
            masks,
            numerator,
            visibility,
            semantic_visibility,
        )
        all_detections.append(
            {
                "view": int(view["index"]),
                "detections": [asdict(detection) for detection in detections],
            }
        )

    scores = numerator / np.maximum(semantic_visibility, 1.0)
    labels = np.argmax(scores, axis=1).astype(np.int32)
    best = np.max(scores, axis=1)
    labels[(visibility == 0) | (best < min_face_score)] = -1
    return labels, scores.astype(np.float32), all_detections
