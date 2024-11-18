import torch
from torch import Tensor


def mask_logsumexp(tens: Tensor, mask: Tensor) -> Tensor:
    tens[~mask] = -float('inf')
    return torch.logsumexp(tens, dim=1)
