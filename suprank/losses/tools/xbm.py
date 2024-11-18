# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.
from typing import Tuple, Type

import torch
import torch.nn as nn
from torch import Tensor

NoneType = Type[None]


class XBM(nn.Module):

    def __init__(
        self,
        queue_size: int,
        embedding_size: int,
        target_size: int,
        factor: float = 1.0,
        factor_true_loss: float = 1.0,
        activate: int = 0,
        get_first: bool = False,
    ) -> NoneType:
        super().__init__()
        self.K = queue_size
        self.embedding_size = embedding_size
        self.target_size = target_size
        self.factor = factor
        self.factor_true_loss = factor_true_loss
        self.activate = activate
        self.get_first = get_first
        self.register_buffer('feats', torch.zeros(self.K, self.embedding_size))
        self.register_buffer('targets', torch.zeros(self.K, self.target_size, dtype=torch.long))
        self.register_buffer('ptr', torch.tensor(0))
        self.register_buffer('total_count', torch.tensor(0))

    @property
    def is_full(self) -> bool:
        return self.total_count >= self.K

    def get(self) -> Tuple[Tensor]:
        if self.is_full:
            return self.feats, self.targets
        else:
            return self.feats[:self.ptr], self.targets[:self.ptr]

    def enqueue_dequeue(self, feats: Tensor, targets: Tensor) -> NoneType:
        q_size = len(targets)
        self.total_count += q_size
        self.total_count.clamp_(max=self.K + 100)
        if self.ptr + q_size > self.K:
            self.feats[-q_size:] = feats
            self.targets[-q_size:] = targets
            self.ptr.fill_(0)
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            self.targets[self.ptr: self.ptr + q_size] = targets
            self.ptr += q_size

    def forward(self, feats: Tensor, targets: Tensor) -> NoneType:
        if self.get_first:
            ft, lb = self.get()
            self.enqueue_dequeue(feats.detach(), targets.detach())
            return ft, lb

        # Original implementation of XBM enqueues before getting.
        # This also ensures that there are positives in the queue
        # when the queue does not cover the entire dataset
        self.enqueue_dequeue(feats.detach(), targets.detach())
        return self.get()

    def extra_repr(self,) -> str:
        repr = ""
        repr += f"    queue_size={self.K},\n"
        repr += f"    embedding_size={self.embedding_size},\n"
        repr += f"    target_size={self.target_size},\n"
        repr += f"    factor={self.factor},\n"
        repr += f"    factor_true_loss={self.factor_true_loss},\n"
        repr += f"    activate={self.activate},"
        return repr
