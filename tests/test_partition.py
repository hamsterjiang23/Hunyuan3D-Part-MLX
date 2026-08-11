from __future__ import annotations

import numpy as np
import trimesh

from split3d.partition import (
    compress_ranges,
    connected_component_labels,
    fill_unlabeled,
    merge_small_label_regions,
    split_instances,
    validate_labels,
)


def test_fill_unlabeled_and_partition_cover_every_face() -> None:
    mesh = trimesh.creation.box()
    labels = np.where(mesh.triangles_center[:, 0] < 0, 0, 1).astype(np.int32)
    labels[0] = -1
    filled = fill_unlabeled(mesh, labels)
    groups = split_instances(mesh, filled)
    assigned = np.concatenate([group.face_indices for group in groups])
    assert len(assigned) == len(mesh.faces)
    assert sorted(assigned.tolist()) == list(range(len(mesh.faces)))


def test_validate_labels_rejects_unknown_part_index() -> None:
    with np.testing.assert_raises(ValueError):
        validate_labels(np.array([0, 2], dtype=np.int32), ["head", "body"], 2)


def test_compress_ranges() -> None:
    assert compress_ranges(np.array([7, 1, 2, 3, 9, 8])) == [[1, 3], [7, 9]]


def test_connected_components_ignore_attribute_seams() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [3, 4, 5]], process=False)
    labels = connected_component_labels(mesh)
    assert labels.tolist() == [0, 0]


def test_merge_small_label_region_into_neighbor() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=1)
    labels = np.zeros(len(mesh.faces), dtype=np.int32)
    labels[0] = 1
    merged = merge_small_label_regions(mesh, labels, min_faces=2)
    assert np.all(merged == 0)


def test_merge_region_relative_to_semantic_size() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=2)
    labels = np.zeros(len(mesh.faces), dtype=np.int32)
    labels[mesh.triangles_center[:, 0] < -0.25] = 1
    island = int(np.argmax(mesh.triangles_center[:, 0]))
    labels[island] = 1
    merged = merge_small_label_regions(mesh, labels, min_faces=1, min_semantic_fraction=0.1)
    assert merged[island] == 0
    assert np.count_nonzero(merged == 1) > 1
