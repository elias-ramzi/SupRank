from typing import List, Mapping, Any, Optional, Type
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, Tensor

NoneType = Type[None]
ArgsType = List[Any]
KwargsType = Mapping[str, Any]


# adapted from :
# https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
class ArcFaceLoss(nn.Module):
    """
    L2 normalize weights and apply temperature scaling on logits.
    """

    takes_ref_embeddings: bool = False

    def __init__(
        self,
        embedding_size: int,
        num_classes: int,
        s: float = 30.0,
        m: float = 0.50,
        ls_eps: float = 0.0,
        easy_margin: bool = False,
        hierarchy_level: Optional[int] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    ) -> NoneType:
        super().__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.easy_margin = easy_margin
        self.hierarchy_level = hierarchy_level
        self.opt = optimizer
        self.sch = scheduler

        self.weight = nn.Parameter(torch.Tensor(num_classes, embedding_size))
        # Initialization from nn.Linear (https://github.com/pytorch/pytorch/blob/v1.0.0/torch/nn/modules/linear.py#L129)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if self.opt is not None:
            self.opt = self.opt(self.parameters())  # partial hydra function
        if self.sch is not None:
            self.sch = self.sch(self.opt)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, embeddings: Tensor, instance_targets: Tensor, **kwargs: KwargsType) -> Tensor:
        if self.hierarchy_level is not None:
            instance_targets = instance_targets[:, self.hierarchy_level]

        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))

        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=cosine.device)
        one_hot.scatter_(1, instance_targets.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.num_classes
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        loss = self.loss_fn(output, instance_targets.long())
        return loss

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
        repr = repr + f"    num_proxies={self.num_classes},\n"
        repr = repr + f"    embedding_size={self.embedding_size},\n"
        repr = repr + f"    margin={self.m},\n"
        repr = repr + f"    scale={self.s},"
        return repr
