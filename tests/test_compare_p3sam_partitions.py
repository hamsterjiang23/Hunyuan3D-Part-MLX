from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_compare_partitions():
    script = Path(__file__).parents[1] / "scripts" / "compare_p3sam_partitions.py"
    spec = importlib.util.spec_from_file_location("compare_p3sam_partitions", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compare_partitions


def test_compare_partitions_is_invariant_to_label_permutation() -> None:
    compare_partitions = _load_compare_partitions()
    cuda = np.asarray([0, 0, 1, 1, 2, 2])
    mlx = np.asarray([8, 8, 4, 4, 7, 7])

    result = compare_partitions(cuda, mlx)

    assert result["adjusted_rand_index"] == pytest.approx(1.0)
    assert result["normalized_mutual_information"] == pytest.approx(1.0)
    assert result["symmetric_best_mask_iou"] == pytest.approx(1.0)
