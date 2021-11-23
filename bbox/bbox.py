from typing import Sequence

import numpy as np


class BBox:
    origin = 1

    def __init__(
        self,
        arr,
        mode: str = "xyxy",
        origin: float = 1,
        copy: bool = False,
    ) -> None:
        """mode = 'xyxy' | 'xywh' | 'ccwh'"""

        if mode not in ("xyxy", "xywh", "ccwh"):
            raise ValueError("mode must be 'xyxy', 'xywh', or 'ccwh'")

        if isinstance(arr, np.ndarray):
            if copy:
                arr = arr.copy()
        else:
            arr = np.array(arr)

        if arr.shape[-1] != 4:
            raise ValueError(
                "The last dimension of the array must be of size 4, "
                "but the given array shape is " + str(arr.shape)
            )

        if mode == "xywh":
            arr[..., 2:] += arr[..., :2] - 1
        elif mode == "ccwh":
            arr[..., :2] -= (arr[..., 2:] - 1) / 2
            arr[..., 2:] += arr[..., :2] - 1

        self._xyxy = arr
        if origin != self.origin:
            self._xyxy += self.origin - origin

    @property
    def shape(self):
        return self._xyxy.shape[:-1]

    @property
    def _x0(self) -> np.ndarray:
        return self._xyxy[..., 0]

    @property
    def _y0(self) -> np.ndarray:
        return self._xyxy[..., 1]

    @property
    def _x1(self) -> np.ndarray:
        return self._xyxy[..., 2]

    @property
    def _y1(self) -> np.ndarray:
        return self._xyxy[..., 3]


def stack(bboxes: Sequence[BBox]) -> BBox:
    arr = np.stack([bbox._xyxy for bbox in bboxes])
    return BBox(arr)
