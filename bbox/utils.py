from typing import Any, Sequence, Union, cast

Pairable = Union[Any, Sequence[Any]]
Pair = tuple[Any, Any]


def pair(x: Pairable) -> Pair:
    try:
        if len(x) == 2:
            return cast(Pair, tuple(x))
        else:
            raise ValueError(f"object length should be 2 but was {len(x)}")
    except TypeError:
        return x, x
