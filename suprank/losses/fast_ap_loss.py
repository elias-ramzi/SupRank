from typing import Mapping, Any, Optional, Tuple, Type

from torch import Tensor
from pytorch_metric_learning import losses

NoneType = Type[None]
KwargsType = Mapping[str, Any]


class FastAPLoss(losses.FastAPLoss):

    takes_ref_embeddings: bool = False  # TODO: this should be possible; investigate PML

    def __init__(self, hierarchy_level: int = 0, **kwargs: KwargsType) -> NoneType:
        super().__init__(**kwargs)
        self.hierarchy_level = hierarchy_level

    def forward(
        self,
        embeddings: Tensor,
        labels: Tensor,
        indices_tuple: Optional[Tuple[Tensor]] = None,
        ref_embeddings: Optional[Tensor] = None,
        ref_labels: Optional[Tensor] = None,
        **kwargs: KwargsType,
    ) -> Tensor:
        # ref_embeddings, ref_labels = ref_embeddings if ref_embeddings is None else embeddings, ref_labels if ref_labels is None else labels
        # if self.hierarchy_level is not None:
        #     labels, ref_labels = labels[:, self.hierarchy_level], ref_labels[:, self.hierarchy_level]

        if self.hierarchy_level is not None:
            labels, ref_labels = labels[:, self.hierarchy_level], ref_labels[:, self.hierarchy_level]

        return super().forward(
            embeddings=embeddings,
            labels=labels,
            indices_tuple=indices_tuple,
        )
