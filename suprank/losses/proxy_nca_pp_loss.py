from typing import List, Mapping, Any, Optional, Type
import torch
import torch.nn.functional as F
from torch import optim, Tensor
import sklearn.preprocessing

NoneType = Type[None]
ArgsType = List[Any]
KwargsType = Mapping[str, Any]


def pairwise_distance(a: Tensor, squared: bool = False) -> Tensor:
    """Computes the pairwise distance matrix with numerical stability."""
    pairwise_distances_squared = torch.add(
        a.pow(2).sum(dim=1, keepdim=True).expand(a.size(0), -1),
        torch.t(a).pow(2).sum(dim=0, keepdim=True).expand(a.size(0), -1)
    ) - 2 * (
        torch.mm(a, torch.t(a))
    )

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.clamp(
        pairwise_distances_squared, min=0.0
    )

    # Get the mask where the zero distances are at.
    error_mask = torch.le(pairwise_distances_squared, 0.0)
    # print(error_mask.sum())
    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = torch.sqrt(
            pairwise_distances_squared + error_mask.float() * 1e-16
        )

    # Undo conditionally adding 1e-16.
    pairwise_distances = torch.mul(
        pairwise_distances,
        (error_mask == False).float()  # noqa: E712
    )

    # Explicitly set diagonals to zero.
    mask_offdiagonals = 1 - torch.eye(
        *pairwise_distances.size(),
        device=pairwise_distances.device
    )
    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)

    return pairwise_distances


def binarize_and_smooth_labels(T: Tensor, nb_classes: int, smoothing_const: float = 0) -> Tensor:
    device = T.device
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(
        T, classes=range(0, nb_classes)
    )
    T = T * (1 - smoothing_const)
    T[T == 0] = smoothing_const / (nb_classes - 1)
    T = torch.FloatTensor(T).to(device)
    return T


class ProxyNCAppLoss(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_size: int,
        scale: float = 3.,
        hierarchy_level: Optional[int] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    ) -> NoneType:
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.scale = scale
        self.hierarchy_level = hierarchy_level
        self.proxies = torch.nn.Parameter(torch.randn(num_classes, embedding_size) / 8)
        self.opt = optimizer
        self.sch = scheduler

        if self.opt is not None:
            self.opt = self.opt(self.parameters())  # partial hydra function
        if self.sch is not None:
            self.sch = self.sch(self.opt)

    def forward(self, X: Tensor, T: Tensor, **kwargs: KwargsType) -> Tensor:
        if self.hierarchy_level is not None:
            T = T[:, self.hierarchy_level]

        P = self.proxies
        # note: self.scale is equal to sqrt(1/T)
        # in the paper T = 1/9, therefore, scale = sart(1/(1/9)) = sqrt(9) = 3
        #  we need to apply sqrt because the pairwise distance is calculated as norm^2

        P = self.scale * F.normalize(P, p=2, dim=-1)
        X = self.scale * F.normalize(X, p=2, dim=-1)

        D = pairwise_distance(
            torch.cat(
                [X, P]
            ),
            squared=True
        )[:X.size()[0], X.size()[0]:]

        # P = F.normalize(P, p=2, dim=-1)
        # X = F.normalize(X, p=2, dim=-1)
        # D = (2 * (1 - (X @ P.t()))) * self.scale

        T = binarize_and_smooth_labels(
            T=T, nb_classes=len(P), smoothing_const=0
        )

        loss = torch.sum(- T * F.log_softmax(-D, -1), -1)
        loss = loss.mean()
        return loss

    def update(self, scaler: Optional[torch.cuda.amp.GradScaler] = None) -> None:
        scaler.step(self.opt)

    def on_epoch(self,) -> NoneType:
        if self.sch:
            self.sch.step()

    def state_dict(self, *args: ArgsType, **kwargs: KwargsType) -> Mapping[str, Any]:
        state = {"super": super().state_dict(*args, **kwargs)}
        state["opt"] = self.opt.state_dict()
        state["sch"] = self.sch.state_dict() if self.sch else None
        return state

    def load_state_dict(self, state_dict: Mapping[str, Any], override: bool = False, *args: ArgsType, **kwargs: KwargsType) -> NoneType:
        super().load_state_dict(state_dict["super"], *args, **kwargs)
        if not override:
            self.opt.load_state_dict(state_dict["opt"])
            if self.sch:
                self.sch.load_state_dict(state_dict["sch"])

    def extra_repr(self,) -> str:
        repr = ""
        repr = repr + f"    scale={self.scale},\n"
        repr = repr + f"    num_classes={self.num_classes},\n"
        repr = repr + f"    embedding_size={self.embedding_size},\n"
        repr = repr + f"    hierarchy_level={self.hierarchy_level},"
        return repr


if __name__ == '__main__':
    di = torch.randn(128, 32)
    labels = torch.randint(10, (128, 1))

    crit = ProxyNCAppLoss(10, 32, 1., hierarchy_level=0)
    loss = crit(di, labels)
