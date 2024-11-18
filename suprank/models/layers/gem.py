from typing import Union, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

NoneType = Type[None]


def gem(x: Tensor, p: Union[float, Tensor] = 3, eps: float = 1e-6) -> Tensor:
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(nn.Module):

    def __init__(self, p: float = 3, learn_p: bool = False, eps: float = 1e-6) -> NoneType:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p).requires_grad_(learn_p)
        self.learn_p = learn_p
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return gem(x, p=self.p, eps=self.eps)

    def extra_repr(self) -> str:
        return 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + f"learn_p={self.learn_p}, " + 'eps=' + str(self.eps)
