from typing import Mapping, Any, Callable

import torch
import torch.nn as nn
from torch import Tensor

from suprank.losses.tools.avg_non_zero_reducer import avg_non_zero_reducer
from suprank.losses.tools.differentiable_rank import DifferentiableRank, Rank, SmoothRank, SupRank, SupHRank, tau_sigmoid
from suprank.losses.tools.pos_mixup import pos_mixup
from suprank.losses.tools.xbm import XBM

KwargsType = Mapping[str, Any]


__all__ = [
    'avg_non_zero_reducer',
    'DifferentiableRank', 'Rank', 'SmoothRank', 'SupRank', 'SupHRank', 'tau_sigmoid',
    'pos_mixup',
    'XBM',
]


REDUCE_DICT = {
    'none': torch.nn.Identity(),
    'mean': torch.mean,
    'sum': torch.sum,
    'avg_non_zero': avg_non_zero_reducer,
}


def reduce(tens: Tensor, reduce_type: str = 'mean') -> Callable:
    return REDUCE_DICT[reduce_type](tens)


def get_differentiable_rank(mode: str, **kwargs: KwargsType) -> nn.Module:
    mode = mode.lower()
    if mode == 'rank':
        return Rank(**kwargs)
    elif mode == 'smoothrank':
        return SmoothRank(**kwargs)
    elif mode == 'suprank':
        return SupRank(**kwargs)
    elif mode == 'suphrank':
        return SupHRank(**kwargs)
    else:
        raise ValueError(f'Rank type: {mode} not supported')
