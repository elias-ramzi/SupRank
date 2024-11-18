from typing import List, Mapping, Any, Optional, Tuple, Type

import torch
from torch import Tensor
from pytorch_metric_learning import losses, distances
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

NoneType = Type[None]
ArgsType = List[Any]
KwargsType = Mapping[str, Any]


class CalibrationLoss(losses.ContrastiveLoss):

    takes_ref_embeddings: bool = True

    def __init__(self, hierarchy_level: Optional[int] = None, *args: ArgsType, **kwargs: KwargsType) -> NoneType:
        super().__init__(*args, **kwargs)
        self.hierarchy_level = hierarchy_level

    def get_default_distance(self) -> distances.BaseDistance:
        return distances.DotProductSimilarity()

    def forward(
        self,
        embeddings: Tensor,
        labels: Tensor,
        ref_embeddings: Optional[Tensor] = None,
        ref_labels: Optional[Tensor] = None,
        **kwargs: KwargsType,
    ) -> Tensor:
        if ref_embeddings is None:
            if self.hierarchy_level is not None:
                labels = labels[:, self.hierarchy_level]
            return super().forward(embeddings, labels)

        if self.hierarchy_level is not None:
            labels = labels[:, self.hierarchy_level]
            ref_labels = ref_labels[:, self.hierarchy_level]

        indices_tuple = self.create_indices_tuple(
            embeddings.size(0),
            embeddings,
            labels,
            ref_embeddings,
            ref_labels,
        )

        combined_embeddings = torch.cat([embeddings, ref_embeddings], dim=0)
        combined_labels = torch.cat([labels, ref_labels], dim=0)
        return super().forward(combined_embeddings, combined_labels, indices_tuple)

    def create_indices_tuple(
        self,
        batch_size: int,
        embeddings: Tensor,
        labels: Tensor,
        E_mem: Tensor,
        L_mem: Tensor,
    ) -> Tuple[Tensor]:
        indices_tuple = lmu.get_all_pairs_indices(labels, L_mem)
        indices_tuple = c_f.shift_indices_tuple(indices_tuple, batch_size)
        return indices_tuple
