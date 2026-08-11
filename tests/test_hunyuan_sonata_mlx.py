from __future__ import annotations

import numpy as np

from split3d.hunyuan.sonata_mlx import attention_padding_maps


def test_attention_padding_matches_released_sonata_layout() -> None:
    pad, unpad, boundaries = attention_padding_maps(np.asarray([3, 8, 18]), patch_size=4)

    np.testing.assert_array_equal(pad[:3], [0, 1, 2])
    np.testing.assert_array_equal(pad[3:11], [3, 4, 5, 6, 7, 4, 5, 6])
    np.testing.assert_array_equal(pad[11:], [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 14, 15])
    np.testing.assert_array_equal(unpad, [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    np.testing.assert_array_equal(boundaries, [0, 3, 7, 11, 15, 19, 23])
