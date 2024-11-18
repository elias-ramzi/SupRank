from typing import List, Mapping, Any, Optional, Tuple, Type

from torch import Tensor
from pytorch_metric_learning import losses, distances, miners

NoneType = Type[None]
ArgsType = List[Any]
KwargsType = Mapping[str, Any]


class TripletLoss(losses.TripletMarginLoss):

    takes_ref_embeddings: bool = True

    def __init__(
        self,
        hierarchy_level: int = None,
        semihard: bool = False,
        *args: ArgsType,
        **kwargs: KwargsType,
    ) -> NoneType:
        super().__init__(*args, **kwargs)
        self.hierarchy_level = hierarchy_level
        self.semihard = semihard

        if self.semihard:
            self.miner = miners.BatchEasyHardMiner()

    def get_default_distance(self,) -> distances.BaseDistance:
        return distances.DotProductSimilarity()

    def forward(
        self,
        embeddings: Tensor,
        labels: Tensor,
        ref_embeddings: Optional[Tensor] = None,
        ref_labels: Optional[Tensor] = None,
        indices_tuple: Optional[Tuple[Tensor]] = None,
        **kwargs: KwargsType,
    ) -> Tensor:
        ref_embeddings, ref_labels = ref_embeddings if ref_embeddings is not None else embeddings, ref_labels if ref_labels is not None else labels
        if self.hierarchy_level is not None:
            labels, ref_labels = labels[:, self.hierarchy_level], ref_labels[:, self.hierarchy_level]

        if self.semihard:
            indices_tuple = self.miner(embeddings, labels, ref_embeddings, ref_labels)

        return super().forward(embeddings, labels, indices_tuple, ref_embeddings, ref_labels)
