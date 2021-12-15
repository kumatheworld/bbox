def pair(x):
    try:
        if len(x) == 2:
            return tuple(x)
        else:
            raise ValueError(f"object length should be 2 but was {len(x)}")
    except TypeError:
        return x, x
