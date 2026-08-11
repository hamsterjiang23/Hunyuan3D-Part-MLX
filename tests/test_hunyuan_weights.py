from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from split3d.hunyuan.weights import inspect_safetensors


def _write_safetensors(path: Path, *, truncate_data: bool = False) -> None:
    header = {
        "linear.bias": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        "linear.weight": {"dtype": "F16", "shape": [2, 3], "data_offsets": [8, 20]},
        "__metadata__": {"format": "pt"},
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    data = bytes(12 if truncate_data else 20)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


def test_inspect_safetensors_reports_shapes_and_completion(tmp_path: Path) -> None:
    path = tmp_path / "weights.safetensors"
    _write_safetensors(path)

    manifest = inspect_safetensors(path)

    assert manifest.complete is True
    assert manifest.expected_file_bytes == path.stat().st_size
    assert [tensor.name for tensor in manifest.tensors] == ["linear.bias", "linear.weight"]
    assert manifest.summary()["dtype_counts"] == {"F16": 1, "F32": 1}


def test_inspect_safetensors_can_read_partial_file_header(tmp_path: Path) -> None:
    path = tmp_path / "partial.safetensors"
    _write_safetensors(path, truncate_data=True)

    manifest = inspect_safetensors(path, require_complete=False)

    assert manifest.complete is False
    with pytest.raises(ValueError, match="truncated safetensors data"):
        inspect_safetensors(path)
