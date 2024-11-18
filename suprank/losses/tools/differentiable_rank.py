from typing import Optional, Callable, Union, Mapping, Any, Type
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor

import suprank.lib as lib
from suprank.losses.tools.pos_mixup import pos_mixup

NoneType = Type[None]
KwargsType = Mapping[str, Any]


@torch.no_grad()  # this no longer done by later version of torch for the heaviside function (> 1.8)
def heaviside(
    tens: Tensor,
    val: float = 1.,
    target: Optional[Tensor] = None,
    general: Optional[bool] = None,
) -> Tensor:
    return torch.heaviside(tens, values=torch.tensor(val, device=tens.device, dtype=tens.dtype))


def tau_sigmoid(
    tens: Tensor,
    tau: float = 0.01,
    target: Optional[Tensor] = None,
    general: Optional[bool] = None,
) -> Tensor:
    exponent = -tens / tau
    exponent = 1. + exponent.clamp(-50, 50).exp()  # clamp the input tensor for stability
    return (1.0 / exponent).type(tens.dtype)


def log2p(tens: Tensor) -> Tensor:
    return torch.log2(1 + tens)


def step_rank(
    tens: Tensor,
    target: Tensor,
    tau: float = 0.01,
    rho: float = 100.,
    delta: float = 0.05,
    start: float = 0.5,
    offset: Optional[float] = None,
    general: bool = False,
) -> Tensor:
    device = tens.device
    dtype = tens.dtype

    if general:
        inferior_relevance = target.view(1, -1) < target[target.bool()].view(-1, 1)
    else:
        inferior_relevance = target.unsqueeze(target.ndim) > target.unsqueeze(target.ndim - 1)

    tens[~inferior_relevance] = heaviside(tens[~inferior_relevance])

    if rho == -1:  # case where we use the sigmoid only for rank minus
        tens[inferior_relevance] = tau_sigmoid(tens[inferior_relevance], tau).type(dtype)
    else:
        pos_mask = (tens > 0).bool()
        if delta is None:  # linear case, when there is no sigmoid at the begining
            tens[inferior_relevance & pos_mask] = rho * tens[inferior_relevance & pos_mask] + offset
        else:
            if offset is None:  # we compute off base on the value of the sigmoid at delta
                offset = tau_sigmoid(torch.tensor([delta], device=device), tau).type(dtype) + start

            margin_mask = tens > delta
            tens[inferior_relevance & pos_mask & ~margin_mask] = start + tau_sigmoid(tens[inferior_relevance & pos_mask & ~margin_mask], tau).type(dtype)
            tens[inferior_relevance & pos_mask & margin_mask] = rho * (tens[inferior_relevance & pos_mask & margin_mask] - delta) + offset

        tens[inferior_relevance & (~pos_mask)] = tau_sigmoid(tens[inferior_relevance & (~pos_mask)], tau).type(dtype)

    return tens


def step_hrank(tens: Tensor, target: Tensor, beta: float = None, leak: float = None, gamma: float = None, general: bool = False) -> Tensor:
    # dtype = tens.dtypes

    # inferior and equal relevance
    if general:
        inferior_relevance = target.view(1, -1) <= target[target.bool()].view(-1, 1)
    else:
        inferior_relevance = target.unsqueeze(target.ndim) >= target.unsqueeze(target.ndim - 1)

    tens[inferior_relevance] = heaviside(tens[inferior_relevance])

    # superior relevance
    superior_relevance = ~inferior_relevance
    pos_mask = tens > 0

    tens[superior_relevance & ~pos_mask] *= leak
    if gamma > 1.0:
        tens[superior_relevance & pos_mask] = heaviside(tens[superior_relevance & pos_mask])
    else:
        tens[superior_relevance & pos_mask] = torch.clamp(beta * tens[superior_relevance & pos_mask] + gamma, max=1.)
    return tens


class DifferentiableRank(nn.Module):

    def __init__(
        self,
        rank_approximation: Callable,
        rank_plus_approximation: Callable,
        hierarchy_level: Union[int, str],
        rank_type: str = 'default',
        reduce_rank: bool = False,
        use_pos_mixup: bool = False,
        force_mixup: bool = False,
        force_general: bool = False,
    ) -> NoneType:
        super().__init__()
        assert rank_type.lower() in ['default', 'ndcg']
        self.rank_approximation = rank_approximation
        self.rank_plus_approximation = rank_plus_approximation
        self.hierarchy_level = hierarchy_level
        self.reduce_rank = reduce_rank
        self.use_pos_mixup = use_pos_mixup
        self.force_mixup = force_mixup
        self.rank_type = rank_type.lower()
        self.force_general = force_general

    def general_forward(self, embeddings: Tensor, labels: Tensor, ref_embeddings: Tensor, ref_labels: Tensor, relevance_fn: Callable) -> Tensor:
        batch_size = embeddings.size(0)
        gallery_size = ref_embeddings.size(0)
        device = embeddings.device
        dtype = embeddings.dtype

        if self.reduce_rank:
            num_pos = (labels == labels[0])
            rank = torch.zeros(batch_size, num_pos, device=device, dtype=dtype)
            rank_plus = torch.zeros(batch_size, num_pos, device=device, dtype=dtype)
        else:
            rank = torch.zeros(batch_size, gallery_size, device=device, dtype=dtype)
            rank_plus = torch.zeros(batch_size, gallery_size, device=device, dtype=dtype)

        normalize_factor = torch.zeros(batch_size, device=device, dtype=dtype)
        for idx in range(batch_size):
            _score = torch.mm(embeddings[idx].view(1, -1), ref_embeddings.t())[0]
            pos_mask = lib.create_label_matrix(
                labels[idx].view(1, -1), ref_labels,
                hierarchy_level=self.hierarchy_level,
                dtype=torch.long,
            )

            # shape M x M
            if self.hierarchy_level == 'MULTI':
                relevances = relevance_fn(pos_mask.view(1, -1)).type(_score.dtype)[0]  # / (labels.size(1) - 1)
            else:
                relevances = pos_mask.view(1, -1).type(_score.dtype)[0]

            min_relevances = torch.where(
                relevances.view(1, -1) < relevances[pos_mask.bool()].view(-1, 1),
                relevances.view(1, -1),
                relevances[pos_mask.bool()].view(-1, 1),
            )

            query = _score.view(1, -1) - _score[pos_mask.bool()].view(-1, 1)
            _rank = self.rank_approximation(torch.clone(query), target=pos_mask, general=True)
            _rank_plus = self.rank_plus_approximation(torch.clone(query), target=pos_mask, general=True)
            mask = torch.ones_like(query, device=device, dtype=torch.bool)
            cond = torch.where(pos_mask)[0]
            mask[(torch.arange(len(cond)), cond)] = 0
            _rank *= mask
            _rank_plus *= mask

            # compute the approximated rank
            _rank = 1. + torch.sum(_rank, dim=-1)

            if self.rank_type == 'default':
                # compute the approximated rank_plus
                _rank_plus = relevances[pos_mask.bool()] + torch.sum(_rank_plus * min_relevances, dim=-1)
            elif self.rank_type == 'ndcg':
                rank = relevances / log2p(rank)
                rank_plus = 1. + relevances.argsort(1, True).argsort(1).float()
                rank_plus = relevances / log2p(rank_plus)

            if self.reduce_rank:
                _rank = _rank[pos_mask].view(-1)
                _rank_plus = _rank_plus[pos_mask].view(-1)

            rank[idx][pos_mask.bool()] += _rank
            rank_plus[idx][pos_mask.bool()] += _rank_plus
            normalize_factor[idx] += relevances.sum()
            del query, cond, mask, relevances, min_relevances, _score, pos_mask
            torch.cuda.empty_cache()

        # shape N
        return rank, rank_plus, normalize_factor

    def quick_forward(self, scores: Tensor, target: Tensor, relevances: Tensor) -> Tensor:
        device = scores.device

        # ------ differentiable ranking of all retrieval set ------
        # compute the mask which ignores difference of scores between an instance and itself
        mask = 1.0 - torch.eye(target.size(1), device=device).unsqueeze(0)
        # compute the difference matrix
        sim_diff = scores.unsqueeze(1) - scores.unsqueeze(1).permute(0, 2, 1)
        min_relevances = torch.where(
            relevances.unsqueeze(1) < relevances.unsqueeze(1).permute(0, 2, 1),
            relevances.unsqueeze(1),
            relevances.unsqueeze(1).permute(0, 2, 1),
        )

        # the relevance is bounded by the maximum (which is one)
        # clone is necessary as rank_approximation is modifying the tensor inplaces
        rank = self.rank_approximation(torch.clone(sim_diff), target=target) * mask
        rank = 1. + torch.sum(rank, dim=-1)

        # compute the approximated Hrank
        if self.rank_type == 'default':
            rank_plus = self.rank_plus_approximation(torch.clone(sim_diff), target=target) * mask
            rank_plus = relevances + torch.sum(rank_plus * min_relevances, dim=-1)
        elif self.rank_type == 'ndcg':
            rank = ((2 ** target) - 1) / log2p(rank)
            rank_plus = 1. + target.argsort(1, True).argsort(1).float()
            rank_plus = ((2 ** target) - 1) / log2p(rank_plus)

        if self.reduce_rank:
            rank = rank[target.bool()].view(rank.size(0), -1)
            rank_plus = rank_plus[target.bool()].view(rank.size(0), -1)

        return rank, rank_plus

    def forward(
        self,
        embeddings: Tensor,
        labels: Tensor,
        relevance_fn: Optional[Callable] = None,
        ref_embeddings: Optional[Tensor] = None,
        ref_labels: Optional[Tensor] = None,
        force_general: bool = False,
        **kwargs: KwargsType,
    ) -> Tensor:
        ref_embeddings, ref_labels = ref_embeddings if ref_embeddings is not None else embeddings, ref_labels if ref_labels is not None else labels
        if self.hierarchy_level != "MULTI":
            labels, ref_labels = labels[:, self.hierarchy_level:self.hierarchy_level + 1], ref_labels[:, self.hierarchy_level:self.hierarchy_level + 1]
        assert (embeddings.size(1) == ref_embeddings.size(1)) and (labels.size(1) == ref_labels.size(1))

        if not (force_general or self.force_general):
            scores = torch.mm(embeddings, ref_embeddings.t())
            target = lib.create_label_matrix(labels, ref_labels, dtype=torch.int64)

            if self.use_pos_mixup:
                scores, target = pos_mixup(scores, target, force_mixup=self.force_mixup)

            if self.hierarchy_level == 'MULTI':
                relevances = relevance_fn(target).type(scores.dtype)
            else:
                relevances = target
            rank, rank_plus = self.quick_forward(scores, target, relevances)
            normalize_factor = relevances.sum(-1)
        else:
            rank, rank_plus, normalize_factor = self.general_forward(embeddings, labels, ref_embeddings, ref_labels, relevance_fn)

        return rank, rank_plus, normalize_factor


class Rank(DifferentiableRank):

    def __init__(self, **kwargs: KwargsType) -> NoneType:
        super().__init__(heaviside, heaviside, **kwargs)


class SmoothRank(DifferentiableRank):

    def __init__(
        self,
        tau: float = 0.01,
        mode: str = 'full',
        **kwargs: KwargsType,
    ) -> NoneType:
        assert mode in ['full', 'no_rank_plus', 'rank_minus_only']
        self.tau = tau
        rank_approximation = partial(tau_sigmoid, tau=tau) if mode in ['full', 'no_rank_plus'] else partial(step_rank, tau=tau, rho=-1, offset=None, delta=None, start=None)
        rank_plus_approximation = partial(tau_sigmoid, tau=tau) if 'full' else heaviside
        super().__init__(
            rank_approximation=rank_approximation,
            rank_plus_approximation=rank_plus_approximation,
            **kwargs,
        )

    def extra_repr(self) -> str:
        return f"tau={self.tau}"


class SupRank(DifferentiableRank):

    def __init__(
        self,
        tau: float = 0.01,
        rho: float = 100.,
        delta: float = 0.05,
        start: float = 0.5,
        offset: Optional[float] = None,
        **kwargs: KwargsType,
    ) -> NoneType:
        self.tau = tau
        self.rho = rho
        self.delta = delta
        self.start = start
        self.offset = offset
        rank_approximation = partial(step_rank, tau=tau, rho=rho, delta=delta, start=start, offset=offset)
        super().__init__(
            rank_approximation=rank_approximation,
            rank_plus_approximation=heaviside,
            **kwargs,
        )

    def extra_repr(self,) -> str:
        repr = ""
        repr += f"    tau={self.tau},\n"
        repr += f"    rho={self.rho},\n"
        repr += f"    delta={self.delta},\n"
        repr += f"    start={self.start},\n"
        repr += f"    offset={self.offset},\n"
        repr += f"    pos_mixup={self.use_pos_mixup},"
        return repr


class SupHRank(DifferentiableRank):

    def __init__(
        self,
        tau: float = 0.01,
        rho: float = 100.,
        delta: float = 0.05,
        start: float = 0.5,
        offset: Optional[float] = None,
        beta: Optional[float] = None,
        leak: Optional[float] = None,
        gamma: Optional[float] = None,
        with_hrank: bool = False,
        **kwargs: KwargsType,
    ) -> NoneType:
        self.tau = tau
        self.rho = rho
        self.delta = delta
        self.start = start
        self.offset = offset
        self.beta = beta
        self.leak = leak
        self.gamma = gamma
        self.with_hrank = with_hrank
        rank_approximation = partial(step_rank, tau=tau, rho=rho, delta=delta, start=start, offset=offset)
        rank_plus_approximation = partial(step_hrank, beta=beta, leak=leak, gamma=gamma) if with_hrank else heaviside
        super().__init__(
            rank_approximation=rank_approximation,
            rank_plus_approximation=rank_plus_approximation,
            **kwargs,
        )

    def extra_repr(self,) -> str:
        repr = ""
        repr += f"    tau={self.tau},\n"
        repr += f"    rho={self.rho},\n"
        repr += f"    delta={self.delta},\n"
        repr += f"    start={self.start},\n"
        repr += f"    offset={self.offset},\n"
        repr += f"    beta={self.beta},\n"
        repr += f"    leak={self.leak},\n"
        repr += f"    gamma={self.gamma},\n"
        repr += f"    with_hrank={self.with_hrank},\n"
        repr += f"    pos_mixup={self.use_pos_mixup},"
        return repr


if __name__ == '__main__':

    embeddings = nn.functional.normalize(torch.randn(32, 128))
    labels = torch.randint(8, (32, 1))
    ref_embeddings = nn.functional.normalize(torch.randn(64, 128))
    ref_labels = torch.randint(8, (64, 1))
    relevance_fn = nn.Identity()

    ranker = Rank(hierarchy_level=0)

    r, r_p, rel = ranker(
        embeddings,
        labels,
        relevance_fn,
        ref_embeddings,
        ref_labels,
    )

    _r, _r_p, rel = ranker(
        embeddings,
        labels,
        relevance_fn,
        ref_embeddings,
        ref_labels,
        force_general=True,
    )

    scores = embeddings @ ref_embeddings.t()
    pos_scores = torch.clone(scores)
    target = lib.create_label_matrix(labels, ref_labels, dtype=torch.int64)
    pos_scores[~target.bool()] = -torch.inf
    g_r = lib.rank(scores)
    g_r_p = lib.rank(pos_scores) * target.bool()
    _g_r = torch.clone(g_r)
    _g_r[~target.bool()] = 0.

    assert (r == g_r).all(), "False quick rank"
    assert (r_p == g_r_p).all(), "False quick rank+"

    assert (_r == _g_r).all(), "False general rank"
    assert (_r_p == g_r_p).all(), "False general rank+"

    rank_diff = SmoothRank(hierarchy_level=0)
    embeddings.requires_grad_(True)

    r, r_p, rel = rank_diff(
        embeddings,
        labels,
        relevance_fn,
        ref_embeddings,
        ref_labels,
    )
    r[~target.bool()] = 0.

    _r, _r_p, rel = rank_diff(
        embeddings,
        labels,
        relevance_fn,
        ref_embeddings,
        ref_labels,
        force_general=True,
    )

    assert torch.allclose(r, _r, atol=1e-3)
    assert torch.allclose(r_p, _r_p, atol=1e-5)

    rank_diff = SupRank(hierarchy_level=0)
    embeddings.requires_grad_(True)

    r, r_p, rel = rank_diff(
        embeddings,
        labels,
        relevance_fn,
        ref_embeddings,
        ref_labels,
    )
    r *= target.bool()
    r.mean().backward()
    grad = torch.clone(embeddings.grad)

    embeddings.grad = None
    embeddings.requires_grad_(True)

    _r, _r_p, rel = rank_diff(
        embeddings,
        labels,
        relevance_fn,
        ref_embeddings,
        ref_labels,
        force_general=True,
    )

    assert torch.allclose(r, _r, atol=1e-3)
    assert torch.allclose(r_p, _r_p, atol=1e-5)

    _r.mean().backward()
    _grad = torch.clone(embeddings.grad)
    torch.allclose(grad, _grad, atol=1e-2)
