import numpy as np


def around(val: float, decimals: int = 4) -> float:
    return np.around(val, decimals=decimals)


def percentage(val: float, decimals: int = 2) -> float:
    return around(val * 100, decimals=decimals)
