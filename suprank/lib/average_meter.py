from typing import Union, Type

import numpy as np
from torch import Tensor

NoneType = Type[None]


def _handle_types(value: Union[Tensor, float]) -> float:
    if hasattr(value, "detach"):
        try:
            value = value.detach().item()
        except ValueError:
            return None

    if np.isnan(value) or np.isinf(value):
        return None

    return value


class AverageMeter:
    def __init__(self,) -> NoneType:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> NoneType:
        val = _handle_types(val)
        if val is not None:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
