from typing import Mapping

from torch.optim import Optimizer


def get_lr(optimizer: Optimizer) -> Mapping[str, float]:
    all_lr = {}
    for i, param_group in enumerate(optimizer.param_groups):
        all_lr[i] = param_group['lr']
    return all_lr
