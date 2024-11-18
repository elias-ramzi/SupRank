from torch import Tensor


def rank(tens: Tensor) -> Tensor:
    return 1 + tens.argsort(1, True).argsort(1)
