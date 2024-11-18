# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in
# https://github.com/msight-tech/research-xbm/blob/master/LICENSE
from typing import Mapping, Any, Optional, Type

import torch
import torch.nn as nn
from torch import Tensor

NoneType = Type[None]
KwargsType = Mapping[str, Any]


class XBMLoss(nn.Module):

    takes_ref_embeddings: bool = True

    def __init__(self, margin: float = 0.5, hierarchy_level: int = 0) -> NoneType:
        super().__init__()
        self.margin = margin
        self.hierarchy_level = hierarchy_level

    def compute_loss(
        self,
        inputs_col: Tensor,
        targets_col: Tensor,
        inputs_row: Tensor,
        target_row: Tensor,
    ) -> Tensor:

        n = inputs_col.size(0)
        # Compute similarity matrix
        sim_mat = torch.matmul(inputs_col, inputs_row.t())
        epsilon = 1e-5
        loss = list()

        neg_count = list()
        for i in range(n):
            pos_pair_ = torch.masked_select(sim_mat[i], targets_col[i] == target_row)
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1 - epsilon)
            neg_pair_ = torch.masked_select(sim_mat[i], targets_col[i] != target_row)

            neg_pair = torch.masked_select(neg_pair_, neg_pair_ > self.margin)

            pos_loss = torch.sum(-pos_pair_ + 1)
            if len(neg_pair) > 0:
                neg_loss = torch.sum(neg_pair)
                neg_count.append(len(neg_pair))
            else:
                neg_loss = 0

            loss.append(pos_loss + neg_loss)

        loss = sum(loss) / n
        return loss

    def forward(
        self,
        embeddings: Tensor,
        labels: Tensor,
        ref_embeddings: Optional[Tensor] = None,
        ref_labels: Optional[Tensor] = None,
        **kwargs: KwargsType,
    ) -> Tensor:
        ref_embeddings, ref_labels = ref_embeddings if ref_embeddings is not None else embeddings, ref_labels if ref_labels is not None else labels
        if self.hierarchy_level is not None:
            labels, ref_labels = labels[:, self.hierarchy_level], ref_labels[:, self.hierarchy_level]

        return self.compute_loss(embeddings, labels, ref_embeddings, ref_labels)

    def extra_repr(self,) -> str:
        return f"margin={self.margin}"
