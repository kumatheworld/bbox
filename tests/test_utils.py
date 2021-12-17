from unittest import TestCase, main

import numpy as np
from bbox.utils import pair


class TestPair(TestCase):
    def assertEqualAsPair(self, x, y):
        self.assertEqual(pair(x), pair(y))

    def test_pair_list(self) -> None:
        x = 0
        y = [0, 0]
        self.assertEqualAsPair(x, y)

    def test_pair_tuple(self) -> None:
        x = 0
        y = (0, 0)
        self.assertEqualAsPair(x, y)

    def test_pair_ndarray(self) -> None:
        x = 0
        y = np.array([0, 0])
        self.assertEqualAsPair(x, y)


if __name__ == "__main__":
    main()
