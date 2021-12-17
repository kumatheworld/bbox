from typing import Any, Sequence, Union


def pair(x: Union[Any, Sequence[Any]]) -> tuple[Any, ...]:
    try:
        if len(x) == 2:
            return tuple(x)
        else:
            raise ValueError(f"object length should be 2 but was {len(x)}")
    except TypeError:
        return x, x
