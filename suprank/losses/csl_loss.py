from typing import List, Mapping, Any, Optional, Union, Type
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, Tensor

import suprank.lib as lib

NoneType = Type[None]
ArgsType = List[Any]
KwargsType = Mapping[str, Any]


class CSLLoss(nn.Module):

    takes_ref_embeddings: bool = False

    def __init__(
        self,
        num_proxies: int,
        margins: List[float] = [0.25, 0.35, 0.45],
        scale: float = 32.0,
        embedding_size: int = 512,
        reduce_type: str = 'sum',
        proxies_seed: int = 0,
        hierarchy_level: Union[str, int] = "MULTI",
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    ) -> NoneType:
        super().__init__()
        self.num_proxies = num_proxies
        self.margins = margins
        self.scale = scale
        self.embedding_size = embedding_size
        self.reduce_type = reduce_type
        self.proxies_seed = proxies_seed
        self.hierarchy_level = hierarchy_level
        self.opt = optimizer
        self.sch = scheduler

        self.init_proxies()

        if self.opt is not None:
            self.opt = self.opt(self.parameters())  # partial hydra function
        if self.sch is not None:
            self.sch = self.sch(self.opt)

        if self.hierarchy_level != 'MULTI':
            assert isinstance(hierarchy_level, int)
            lib.LOGGER.warning(f"Hierarchy_level (={self.hierarchy_level}) was passed for a HAP surrogate loss")

    @lib.get_set_random_state
    def init_proxies(self,) -> NoneType:
        # Init as ProxyNCA++
        lib.random_seed(self.proxies_seed, backend=False)
        self.weight = nn.Parameter(torch.Tensor(self.num_proxies, self.embedding_size))
        # Initialization from nn.Linear (https://github.com/pytorch/pytorch/blob/v1.0.0/torch/nn/modules/linear.py#L129)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, embeddings: Tensor, labels: Tensor, **kwargs: KwargsType) -> Tensor:
        P = F.normalize(self.weight)
        similarities = embeddings @ P.t()

        losses = []
        for lvl in range(labels.size(1)):
            index = labels[:, lvl].unsqueeze(1) == self.labels[:, lvl].unsqueeze(0)
            LSE_neg = lib.mask_logsumexp(similarities * self.scale, ~index)
            if lvl == 0:
                LSE_pos_high = lib.mask_logsumexp(- similarities * self.scale, index)
            lss = F.softplus(LSE_neg + LSE_pos_high + self.margins[lvl] * self.scale).mean()
            losses.append(lss)

        if self.reduce_type == 'sum':
            loss = sum(losses)
        elif self.reduce_type == 'mean':
            loss = sum(losses) / len(losses)
        elif self.reduce_type == 'none':
            loss = losses
        else:
            raise ValueError

        return loss

    def register_labels(self, labels: Tensor) -> NoneType:
        assert len(self.margins) == labels.shape[1]
        assert self.num_proxies == len(labels[:, 0].unique())
        self.labels = nn.Parameter(labels.unique(dim=0), requires_grad=False)
        lib.LOGGER.info(f"Labels registered for {self.__class__.__name__}")

    def update(self, scaler: Optional[torch.cuda.amp.GradScaler] = None) -> NoneType:
        scaler.step(self.opt)

    def on_epoch(self,) -> NoneType:
        if self.sch:
            self.sch.step()

    def state_dict(self, *args: ArgsType, **kwargs: KwargsType) -> Mapping[str, Any]:
        state = {"super": super().state_dict(*args, **kwargs)}
        state["opt"] = self.opt.state_dict()
        state["sch"] = self.sch.state_dict() if self.sch else None
        return state

    def load_state_dict(self, state_dict: Mapping[str, Any], *args: ArgsType, **kwargs: KwargsType) -> NoneType:
        super().load_state_dict(state_dict["super"], *args, **kwargs)
        self.opt.load_state_dict(state_dict["opt"])
        self.sch.load_state_dict(state_dict["sch"])

    def extra_repr(self,) -> str:
        repr = ''
        repr = repr + f"    num_proxies={self.num_proxies},\n"
        repr = repr + f"    embedding_size={self.embedding_size},\n"
        repr = repr + f"    margins={self.margins},\n"
        repr = repr + f"    scale={self.scale},\n"
        repr = repr + f"    reduce_type={self.reduce_type},\n"
        repr = repr + f"    proxies_seed={self.proxies_seed},\n"
        return repr
