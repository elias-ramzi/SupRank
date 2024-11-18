from typing import Mapping, Any, Type

import torch.nn as nn
from torch import Tensor

NoneType = Type[None]
KwargsType = Mapping[str, Any]


class MeanMaxPooling(nn.Module):

    def __init__(self, **kwargs: KwargsType) -> NoneType:
        super().__init__()
        self.mean = nn.AdaptiveAvgPool2d(**kwargs)
        self.max = nn.AdaptiveMaxPool2d(**kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.mean(x) + self.max(x)
