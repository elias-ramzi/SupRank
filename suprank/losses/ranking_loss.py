from typing import Mapping, Any, Callable, Optional, List, Type

import torch
import torch.nn as nn
from torch import Tensor

from suprank.losses.tools import get_differentiable_rank, reduce, tau_sigmoid

NoneType = Type[None]
KwargsType = Mapping[str, Any]


class RankingLoss(nn.Module):

    takes_ref_embeddings: bool = True

    def __init__(
        self,
        rank_mode: str,
        return_type: str = '1-mean',
        reduce_type: str = 'mean',
        **kwargs: KwargsType,
    ) -> NoneType:
        super().__init__()
        assert return_type in ['none', 'mean', '1-', '1-mean']
        self.ranker = get_differentiable_rank(rank_mode, **kwargs)
        self.return_type = return_type
        self.reduce_type = reduce_type
        self.hierarchy_level = self.ranker.hierarchy_level

    def compute_ranking_loss(self, rank: Tensor, rank_plus: Tensor, normalize_factor: Tensor) -> Tensor:
        assert rank.size() == rank_plus.size()
        assert len(rank) == len(normalize_factor)

    def forward(
        self,
        embeddings: Tensor,
        labels: Tensor,
        relevance_fn: Callable,
        ref_embeddings: Optional[Tensor] = None,
        ref_labels: Optional[Tensor] = None,
        force_general: bool = False,
        **kwargs: KwargsType,
    ) -> Tensor:
        rank, rank_plus, normalize_factor = self.ranker(
            embeddings=embeddings,
            labels=labels,
            relevance_fn=relevance_fn,
            ref_embeddings=ref_embeddings,
            ref_labels=ref_labels,
            force_general=force_general,
        )

        ranking_score = self.compute_ranking_loss(rank, rank_plus, normalize_factor)

        if self.return_type == 'none':
            return ranking_score
        elif self.return_type == 'mean':
            return ranking_score.mean()
        elif self.return_type == '1-':
            return 1 - ranking_score
        elif self.return_type == '1-mean':
            return reduce(1 - ranking_score, self.reduce_type)


def compute_ap_loss(rank: Tensor, rank_plus: Tensor, normalize_factor: Tensor) -> Tensor:
    return (rank_plus / rank).sum(-1) / normalize_factor


def compute_recall_loss(rank: Tensor, rank_plus: Tensor, normalize_factor: Tensor, list_at_k: List[int], temp: float = 1.0) -> Tensor:
    device = rank.device
    dtype = rank.dtype

    recall = torch.zeros(len(rank), device=device, dtype=dtype)
    for at_k in list_at_k:
        normalize = torch.where(normalize_factor < at_k, normalize_factor, torch.tensor(at_k, device=device).float())
        recall += tau_sigmoid(at_k - rank, temp).sum(-1) / normalize.float()

    return recall / len(list_at_k)


def compute_ndcg_loss(dcg: Tensor, idcg: Tensor) -> Tensor:
    return dcg.sum(-1) / idcg.sum(-1)


class SmoothAPLoss(RankingLoss):

    def __init__(self, **kwargs: KwargsType) -> NoneType:
        kwargs['rank_mode'] = 'SmoothRank'
        super().__init__(**kwargs)

    def compute_ranking_loss(self, rank: Tensor, rank_plus: Tensor, normalize_factor: Tensor) -> Tensor:
        return compute_ap_loss(rank, rank_plus, normalize_factor)


class SupAPLoss(RankingLoss):

    def __init__(self, **kwargs: KwargsType) -> NoneType:
        kwargs['rank_mode'] = 'SupRank'
        super().__init__(**kwargs)

    def compute_ranking_loss(self, rank: Tensor, rank_plus: Tensor, normalize_factor: Tensor) -> Tensor:
        return compute_ap_loss(rank, rank_plus, normalize_factor)


class SmoothHAPLoss(RankingLoss):

    def __init__(self, **kwargs: KwargsType) -> NoneType:
        kwargs['rank_mode'] = 'SmoothRank'
        super().__init__(**kwargs)

    def compute_ranking_loss(self, rank: Tensor, rank_plus: Tensor, normalize_factor: Tensor) -> Tensor:
        return compute_ap_loss(rank, rank_plus, normalize_factor)


class SupHAPLoss(RankingLoss):

    def __init__(self, **kwargs: KwargsType) -> NoneType:
        kwargs['rank_mode'] = 'SupHRank'
        super().__init__(**kwargs)

    def compute_ranking_loss(self, rank: Tensor, rank_plus: Tensor, normalize_factor: Tensor) -> Tensor:
        return compute_ap_loss(rank, rank_plus, normalize_factor)


class SupRecallLoss(RankingLoss):

    def __init__(self, at_k: List[int], temp: float = 1.0, **kwargs: KwargsType) -> NoneType:
        kwargs['rank_mode'] = 'SupRank'
        kwargs['reduce_rank'] = True
        super().__init__(**kwargs)
        self.at_k = at_k
        self.temp = temp

    def compute_ranking_loss(self, rank: Tensor, rank_plus: Tensor, normalize_factor: Tensor) -> Tensor:
        return compute_recall_loss(rank, rank_plus, normalize_factor, self.at_k, self.temp)


class SmoothRecallLoss(RankingLoss):

    def __init__(self, at_k: List[int], temp: float = 1.0, **kwargs: KwargsType) -> NoneType:
        kwargs['rank_mode'] = 'SmoothRank'
        kwargs['reduce_rank'] = True
        super().__init__(**kwargs)
        self.at_k = at_k
        self.temp = temp

    def compute_ranking_loss(self, rank: Tensor, rank_plus: Tensor, normalize_factor: Tensor) -> Tensor:
        return compute_recall_loss(rank, rank_plus, normalize_factor, self.at_k, self.temp)


class SmoothNDCGLoss(RankingLoss):

    def __init__(self, **kwargs: KwargsType) -> NoneType:
        kwargs['rank_mode'] = 'SmoothRank'
        kwargs['rank_type'] = 'NDCG'
        super().__init__(**kwargs)

    def compute_ranking_loss(self, rank: Tensor, rank_plus: Tensor, normalize_factor: Optional[Tensor] = None) -> Tensor:
        return compute_ndcg_loss(rank, rank_plus)


class SupNDCGLoss(RankingLoss):

    def __init__(self, **kwargs: KwargsType) -> NoneType:
        kwargs['rank_mode'] = 'SupRank'
        kwargs['rank_type'] = 'NDCG'
        super().__init__(**kwargs)

    def compute_ranking_loss(self, rank: Tensor, rank_plus: Tensor, normalize_factor: Optional[Tensor] = None) -> Tensor:
        return compute_ndcg_loss(rank, rank_plus)
