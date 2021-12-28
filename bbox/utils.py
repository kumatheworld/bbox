from collections.abc import Collection
from typing import TypeVar, Union, cast

T = TypeVar("T")
Pairable = Union[T, Collection[T]]
Pair = tuple[T, T]


def pair(x: Pairable[T]) -> Pair[T]:
    try:
        if len(x) == 2:
            return cast(Pair[T], tuple(x))
        else:
            raise ValueError(f"object length should be 2 but was {len(x)}")
    except TypeError:
        return cast(Pair[T], (x, x))
