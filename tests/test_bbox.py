from typing import Union
from unittest import TestCase, main

import numpy as np
from bbox.bbox import BBox, stack
from numpy.testing import assert_array_equal


class TestBBox(TestCase):
    def setUp(self, seed=0) -> None:
        np.random.seed(seed)

    @staticmethod
    def _generate_random_array() -> np.ndarray:
        return np.random.rand(2, 3, 4)

    def test_init_arr_pos(self) -> None:
        valid_arrays = [(0, 0, 0, 0), [[1, 2, 3, 4]], np.random.rand(2, 3, 4)]
        for arr in valid_arrays:
            bbox = BBox(arr)
            self.assertIsInstance(bbox, BBox)

    def test_init_arr_neg(self) -> None:
        invalid_arrays = [(5, 6), [[7], [8], [9], [0]], np.random.rand(4, 5)]
        for arr in invalid_arrays:
            with self.assertRaises(ValueError):
                BBox(arr)

    def test_init_mode_pos(self) -> None:
        arr = self._generate_random_array()
        valid_modes = ("xyxy", "xywh", "ccwh")
        for mode in valid_modes:
            bbox = BBox(arr, mode=mode)
            self.assertIsInstance(bbox, BBox)

    def test_init_mode_neg(self) -> None:
        arr = self._generate_random_array()
        invalid_modes = ("xxyy", "xyhw", "cxcywh")
        for mode in invalid_modes:
            with self.assertRaises(ValueError):
                BBox(arr, mode=mode)

    def test_init_origin(self) -> None:
        arr = self._generate_random_array()
        origins = (0, 1)
        for origin in origins:
            bbox = BBox(arr, origin=origin)
            self.assertIsInstance(bbox, BBox)

    def test_init_copy_false(self) -> None:
        arr = self._generate_random_array()
        bbox = BBox(arr, copy=False)
        self.assertIs(arr, bbox._xyxy)

    def test_init_copy_true(self) -> None:
        arr = self._generate_random_array()
        bbox = BBox(arr, copy=True)
        self.assertIsNot(arr, bbox._xyxy)

    def test_stack_no_axis(self) -> None:
        arr = self._generate_random_array()
        bboxes = [BBox(a) for a in arr]
        bbox = stack(bboxes)
        assert_array_equal(bbox._xyxy, arr)

    def test_stack_pos_pos_axis(self) -> None:
        arr = self._generate_random_array()
        idx: list[Union[slice, int]] = []
        for axis in range(arr.ndim - 1):
            bboxes = [BBox(arr[tuple(idx + [i])]) for i in range(arr.shape[axis])]
            bbox = stack(bboxes, axis)
            assert_array_equal(bbox._xyxy, arr)
            idx.append(slice(None))

if __name__ == "__main__":
    main()
