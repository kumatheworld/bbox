from unittest import TestCase, main

import numpy as np
from bbox.bbox import BBox


class TestBBox(TestCase):
    def setUp(self, seed=0) -> None:
        np.random.seed(seed)

    def test_init_arr_pos(self) -> None:
        valid_arrays = [(0, 0, 0, 0), [[1, 2, 3, 4]], np.random.rand(2, 3, 4)]
        for arr in valid_arrays:
            bbox = BBox(arr)
            self.assertIsInstance(bbox, BBox)

    def test_init_mode_pos(self) -> None:
        arr = np.random.rand(2, 3, 4)
        valid_modes = ("xyxy", "xywh", "ccwh")
        for mode in valid_modes:
            bbox = BBox(arr, mode=mode)
            self.assertIsInstance(bbox, BBox)

    def test_init_mode_neg(self) -> None:
        arr = np.random.rand(2, 3, 4)
        invalid_modes = ("xxyy", "xyhw", "cxcywh")
        for mode in invalid_modes:
            with self.assertRaises(ValueError):
                BBox(arr, mode=mode)


if __name__ == "__main__":
    main()
