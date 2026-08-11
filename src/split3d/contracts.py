from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PartRecord:
    part_id: str
    name: str
    semantic_name: str
    instance_index: int
    confidence: float
    face_count: int
    source_face_ranges: list[list[int]]
    bounds: list[list[float]]
    node_name: str
    file: str | None
    render: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SplitManifest:
    schema_version: int
    source: str
    source_sha256: str
    source_faces: int
    source_vertices: int
    assignment: str
    split_glb: str
    parts: list[PartRecord]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["parts"] = [part.to_dict() for part in self.parts]
        return data
