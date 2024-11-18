from typing import Union, List, Optional

import torch
from torch import Tensor

import suprank.lib as lib


def relevance_for_batch(
    batch: Tensor,
    alpha: Union[float, List[float]] = 1.0,
    check_for: Optional[List[int]] = None,
    type: str = 'pop',
) -> Tensor:
    if type == 'map':
        relevances = get_relevances_map(
            batch,
            weights=alpha,
        )
    elif type == 'pop':
        relevances = get_relevances_pop(
            batch,
            levels=max(check_for),
            alpha=alpha,
        )
    else:
        raise ValueError

    return relevances


def compute_relevance_on_the_fly(
    batch: Tensor,
    alpha: Union[float, List[float]] = 1.0,
    check_for: Optional[List[int]] = None,
    type: str = 'pop',
) -> Tensor:
    relevances = relevance_for_batch(
        batch,
        alpha=alpha,
        check_for=check_for,
        type=type,
    )
    return lib.create_relevance_matrix(batch, relevances)


def get_relevances_map(target: Tensor, weights: List[float]) -> Tensor:
    target = target.long()
    weights = torch.tensor(list(weights), device=target.device)
    level = weights.size(-1)

    counts = torch.zeros(target.size(0), level + 1, device=target.device, dtype=torch.float)
    for i, line in enumerate(target):
        o, c = torch.unique(line, sorted=True, return_counts=True)
        counts[i, o] = c.float()

    counts[:, 0] = 0
    counts = counts + torch.sum(counts, dim=1, keepdims=True) - torch.cumsum(counts, dim=1)
    counts[:, 0] = 0
    counts[:, 1:] = weights / counts[:, 1:]

    return torch.cumsum(counts, dim=1)


def get_relevances_pop(target: Tensor, levels: int, alpha: float = 1.0) -> Tensor:
    target = target.long()
    weights = [(x / (levels - 1)) ** alpha for x in range(levels + 1)]  # list(range(levels+1))
    weights = torch.tensor(weights, device=target.device)

    counts = torch.zeros(target.size(0), levels + 1, device=target.device, dtype=torch.float)
    for i, line in enumerate(target):
        o, c = torch.unique(line, sorted=True, return_counts=True)
        counts[i, o] = (weights[o] / c).float()

    return counts
