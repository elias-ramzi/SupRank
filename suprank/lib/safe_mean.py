from torch import Tensor


def safe_mean(tens: Tensor) -> Tensor:
    if tens.nelement() != 0:
        return tens.mean()
    return tens.detach()
