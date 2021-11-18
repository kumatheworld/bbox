from unittest import TestCase, main

import numpy as np
from bbox.bbox import BBox


class TestBBox(TestCase):
    def test_init(self) -> None:
        arr = np.random.rand(2, 3, 4)
        bbox = BBox(arr)
        self.assertIsInstance(bbox, BBox)


if __name__ == "__main__":
    main()
