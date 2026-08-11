from __future__ import annotations

import json
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]

    @property
    def element_count(self) -> int:
        count = 1
        for dimension in self.shape:
            count *= dimension
        return count

    @property
    def byte_count(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]

    def validate(self) -> None:
        if self.dtype not in _DTYPE_BYTES:
            raise ValueError(f"unsupported safetensors dtype for {self.name}: {self.dtype}")
        if any(dimension < 0 for dimension in self.shape):
            raise ValueError(f"negative tensor dimension for {self.name}: {self.shape}")
        start, end = self.data_offsets
        if start < 0 or end < start:
            raise ValueError(f"invalid data offsets for {self.name}: {self.data_offsets}")
        expected = self.element_count * _DTYPE_BYTES[self.dtype]
        if self.byte_count != expected:
            raise ValueError(
                f"byte count mismatch for {self.name}: header={self.byte_count}, expected={expected}"
            )


@dataclass(frozen=True)
class SafetensorManifest:
    path: Path
    header_bytes: int
    file_bytes: int
    tensors: tuple[TensorSpec, ...]
    metadata: dict[str, str]

    @property
    def data_bytes(self) -> int:
        return max((tensor.data_offsets[1] for tensor in self.tensors), default=0)

    @property
    def expected_file_bytes(self) -> int:
        return 8 + self.header_bytes + self.data_bytes

    @property
    def complete(self) -> bool:
        return self.file_bytes >= self.expected_file_bytes

    def summary(self) -> dict[str, Any]:
        dtype_counts = Counter(tensor.dtype for tensor in self.tensors)
        prefix_counts = Counter(tensor.name.split(".", 1)[0] for tensor in self.tensors)
        return {
            "path": str(self.path),
            "file_bytes": self.file_bytes,
            "expected_file_bytes": self.expected_file_bytes,
            "complete": self.complete,
            "tensor_count": len(self.tensors),
            "dtype_counts": dict(sorted(dtype_counts.items())),
            "top_level_prefix_counts": dict(sorted(prefix_counts.items())),
            "metadata": self.metadata,
        }


def inspect_safetensors(path: Path, *, require_complete: bool = True) -> SafetensorManifest:
    path = path.expanduser().resolve()
    file_bytes = path.stat().st_size
    with path.open("rb") as handle:
        header_size_raw = handle.read(8)
        if len(header_size_raw) != 8:
            raise ValueError(f"not a safetensors file (missing header length): {path}")
        (header_bytes,) = struct.unpack("<Q", header_size_raw)
        if header_bytes <= 1 or header_bytes > 1 << 30:
            raise ValueError(f"invalid safetensors header length {header_bytes}: {path}")
        header_raw = handle.read(header_bytes)
        if len(header_raw) != header_bytes:
            raise ValueError(f"truncated safetensors header: {path}")

    try:
        header = json.loads(header_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid safetensors JSON header: {path}") from error
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header must be an object: {path}")

    metadata_raw = header.pop("__metadata__", {})
    if not isinstance(metadata_raw, dict):
        raise ValueError(f"safetensors metadata must be an object: {path}")
    metadata = {str(key): str(value) for key, value in metadata_raw.items()}

    tensors: list[TensorSpec] = []
    for name, value in header.items():
        if not isinstance(value, dict):
            raise ValueError(f"invalid tensor entry for {name}: {path}")
        try:
            tensor = TensorSpec(
                name=name,
                dtype=str(value["dtype"]),
                shape=tuple(int(dimension) for dimension in value["shape"]),
                data_offsets=tuple(int(offset) for offset in value["data_offsets"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid tensor metadata for {name}: {path}") from error
        if len(tensor.data_offsets) != 2:
            raise ValueError(f"invalid data offset rank for {name}: {tensor.data_offsets}")
        tensor.validate()
        tensors.append(tensor)

    tensors.sort(key=lambda tensor: tensor.data_offsets)
    previous_end = 0
    for tensor in tensors:
        if tensor.data_offsets[0] != previous_end:
            raise ValueError(
                f"non-contiguous tensor data before {tensor.name}: "
                f"expected offset {previous_end}, got {tensor.data_offsets[0]}"
            )
        previous_end = tensor.data_offsets[1]

    manifest = SafetensorManifest(
        path=path,
        header_bytes=header_bytes,
        file_bytes=file_bytes,
        tensors=tuple(tensors),
        metadata=metadata,
    )
    if require_complete and not manifest.complete:
        raise ValueError(
            f"truncated safetensors data: {path} has {file_bytes} bytes, "
            f"expected {manifest.expected_file_bytes}"
        )
    return manifest
