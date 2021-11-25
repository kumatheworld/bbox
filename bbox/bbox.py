from re import sub
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


def stack(bboxes: Sequence[BBox], axis=0) -> BBox:
    if axis < 0:
        axis -= 1
    arr = np.stack([bbox._xyxy for bbox in bboxes], axis)
    return BBox(arr)


def loadtxt(
    fname,
    mode="xywh",
    origin=1,
    dtype=float,
    delimiter=",",
    usecols=None,
    ndmin=2,
    start=None,
    stop=None,
    step=None,
):
    if delimiter:
        arr = np.loadtxt(
            fname, dtype, delimiter=delimiter, usecols=usecols, ndmin=ndmin
        )
    else:
        with open(fname) as f:
            arr_str = [sub(",|\t", " ", line) for line in f]
        arr = np.loadtxt(arr_str, dtype, usecols=usecols, ndmin=ndmin)
    arr = arr[start:stop:step]
    return BBox(arr, mode, origin)
