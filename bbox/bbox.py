from collections.abc import Iterable
from copy import copy
from os import PathLike
from typing import Literal, Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike

from bbox.utils import Pairable, pair


class BBox:
    def __init__(
        self,
        arr: ArrayLike,
        mode: Literal["xyxy", "xywh", "ccwh"] = "xyxy",
        base: float = 0,
        copy: bool = False,
    ) -> None:
        if mode not in ("xyxy", "xywh", "ccwh"):
            raise ValueError("mode must be 'xyxy', 'xywh', or 'ccwh'")

        if isinstance(arr, np.ndarray):
            if copy:
                arr = arr.copy()
        else:
            arr = np.array(arr)

        if arr.shape[-1] != 4:
            raise ValueError("expected array of shape (*, 4) but got " + str(arr.shape))

        if mode == "xywh":
            arr[..., 2:] += arr[..., :2] - 1
        elif mode == "ccwh":
            arr[..., :2] -= (arr[..., 2:] - 1) / 2
            arr[..., 2:] += arr[..., :2] - 1

        arr -= base
        self._xyxy = arr

    @property
    def ndim(self) -> int:
        return self._xyxy.ndim - 1

    @property
    def shape(self) -> tuple[int, ...]:
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

    @property
    def _center(self) -> np.ndarray:
        return (self._xyxy[..., :2] + self._xyxy[..., 2:]) / 2

    @property
    def center(self) -> tuple[np.ndarray, np.ndarray]:
        cxcy = self._center
        return cxcy[..., 0], cxcy[..., 1]

    @property
    def _size(self) -> np.ndarray:
        return self._xyxy[..., 2:] - self._xyxy[..., :2] + 1

    @property
    def size(self) -> tuple[np.ndarray, np.ndarray]:
        wh = self._size
        return wh[..., 0], wh[..., 1]

    @property
    def area(self) -> np.ndarray:
        width, height = self.size
        return width * height * self.is_valid()

    def __repr__(self) -> str:
        return f"BBox({repr(self._xyxy)})"

    def __str__(self) -> str:
        return f"BBox(x0={self._x0}, y0={self._y0}, x1={self._x1}, y1={self._y1})"

    def __copy__(self) -> "BBox":
        return BBox(self._xyxy, copy=True)

    def __len__(self) -> int:
        return self.shape[0]

    # TODO: accept negative indices and tuples of ellipses
    def __getitem__(self, key: Union[int, slice]) -> "BBox":
        return BBox(self._xyxy[key])

    def __setitem__(self, key: Union[int, slice], value: np.ndarray) -> None:
        self._xyxy[key] = value

    def __and__(self, other: "BBox") -> "BBox":
        lt = np.maximum(self._xyxy[..., :2], other._xyxy[..., :2])
        rb = np.minimum(self._xyxy[..., 2:], other._xyxy[..., 2:])
        return BBox(np.concatenate((lt, rb), -1))

    def __round__(self) -> "BBox":
        return BBox(self._xyxy.round())

    def __iadd__(self, point: Pairable[float]) -> "BBox":
        dx, dy = pair(point)
        self._xyxy[..., 0::2] += dx
        self._xyxy[..., 1::2] += dy
        return self

    def __add__(self, point: Pairable[float]) -> "BBox":
        bb = copy(self)
        bb += point
        return bb

    def __isub__(self, point: Pairable[float]) -> "BBox":
        dx, dy = pair(point)
        self._xyxy[..., 0::2] -= dx
        self._xyxy[..., 1::2] -= dy
        return self

    def __sub__(self, point: Pairable[float]) -> "BBox":
        bb = copy(self)
        bb -= point
        return bb

    def __imul__(self, factor: Pairable[float]) -> "BBox":
        self.scale(factor, -0.5)
        return self

    def __mul__(self, factor: Pairable[float]) -> "BBox":
        bb = copy(self)
        bb *= factor
        return bb

    def __itruediv__(self, factor: Pairable[float]) -> "BBox":
        kx, ky = pair(factor)
        self *= (1 / kx, 1 / ky)
        return self

    def __truediv__(self, factor: Pairable[float]) -> "BBox":
        bb = copy(self)
        bb /= factor
        return bb

    def is_valid(self) -> np.ndarray:
        w, h = self.size
        return (w > 0) & (h > 0)

    def scale(
        self, factor: Pairable[float], fixed_point: Pairable[float], inplace=True
    ) -> "BBox":
        kx, ky = pair(factor)
        fpx, fpy = pair(fixed_point)
        bb = self if inplace else copy(self)
        bb._xyxy[..., 0] = (bb._xyxy[..., 0] - 0.5 - fpx) * kx + fpx + 0.5
        bb._xyxy[..., 1] = (bb._xyxy[..., 1] - 0.5 - fpy) * ky + fpy + 0.5
        bb._xyxy[..., 2] = (bb._xyxy[..., 2] + 0.5 - fpx) * kx + fpx - 0.5
        bb._xyxy[..., 3] = (bb._xyxy[..., 3] + 0.5 - fpy) * ky + fpy - 0.5
        return bb

    def reshape(self, *shape: int) -> "BBox":
        arr = self._xyxy.reshape(*shape, 4)
        return BBox(arr)

    def is_inside(self, im_size: Pairable) -> np.ndarray:
        w, h = pair(im_size)
        x0 = self._x0
        y0 = self._y0
        x1 = self._x1
        y1 = self._y1
        return (
            (0 <= x0)
            & (x0 <= w - 1)
            & (0 <= y0)
            & (y0 <= h - 1)
            & (0 <= x1)
            & (x1 <= w - 1)
            & (0 <= y1)
            & (y1 <= h - 1)
        )

    def rectify(self, im_size: Pairable) -> "BBox":
        w, h = pair(im_size)
        xx = self._xyxy[..., 0::2].clip(0, w - 1)
        yy = self._xyxy[..., 1::2].clip(0, h - 1)
        arr = np.stack((xx[..., 0], yy[..., 0], xx[..., 1], yy[..., 1]), -1)
        return BBox(arr)

    def IoU(self, other: "BBox") -> float:
        its_area = (self & other).area
        return its_area / (self.area + other.area - its_area)


class AxisError(ValueError, IndexError):
    def __init__(self, axis: int, ndim: int) -> None:
        super().__init__(f"axis {axis} is out of bounds for BBox of dimension {ndim}")


def stack(bboxes: Sequence[BBox], axis: int = 0) -> BBox:
    ndim = bboxes[0].ndim
    if axis not in range(-ndim - 1, ndim + 1):
        raise AxisError(axis, ndim)
    if axis < 0:
        axis -= 1
    arr = np.stack([bbox._xyxy for bbox in bboxes], axis)
    return BBox(arr)


def loadtxt(
    fname: Union[str, PathLike, Iterable[str], Iterable[bytes]],
    mode: Literal["xyxy", "xywh", "ccwh"] = "xywh",
    base: float = 1,
    dtype: type = float,
    delimiter: Optional[str] = ",",
    ndmin: Literal[0, 1, 2] = 2,
    **np_loadtxt_kwargs,
):
    arr = np.loadtxt(
        fname, dtype, delimiter=delimiter, ndmin=ndmin, **np_loadtxt_kwargs
    )
    return BBox(arr, mode, base)
