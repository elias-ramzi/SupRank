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
# https://github.com/azgo14/classification_metric_learning/blob/master/metric_learning/modules/losses.py
class ClusterLoss(nn.Module):
    """
    L2 normalize weights and apply temperature scaling on logits.
    """

    takes_ref_embeddings: bool = False

    def __init__(
        self,
        embedding_size: int,
        num_classes: int,
        num_centers: int = 1,
        temperature: float = 0.05,
        temperature_centers: float = 0.1,
        hierarchy_level: Optional[int] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    ) -> NoneType:
        super().__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.num_centers = num_centers
        self.temperature = temperature
        self.temperature_centers = temperature_centers
        self.hierarchy_level = hierarchy_level
        self.opt = optimizer
        self.sch = scheduler

        self.weight = nn.Parameter(torch.Tensor(num_classes * num_centers, embedding_size))
        # Initialization from nn.Linear (https://github.com/pytorch/pytorch/blob/v1.0.0/torch/nn/modules/linear.py#L129)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if self.opt is not None:
            self.opt = self.opt(self.parameters())  # partial hydra function
        if self.sch is not None:
            self.sch = self.sch(self.opt)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, embeddings: Tensor, instance_targets: Tensor, **kwargs: KwargsType) -> Tensor:
        if self.hierarchy_level is not None:
            instance_targets = instance_targets[:, self.hierarchy_level]

        norm_weight = F.normalize(self.weight, dim=1)

        prediction_logits = F.linear(embeddings, norm_weight)

        if self.num_centers > 1:
            prediction_logits = prediction_logits.reshape(embeddings.size(0), self.num_classes, self.num_centers)
            prob = F.softmax(prediction_logits / self.temperature_centers, dim=2)
            prediction_logits = (prediction_logits * prob).sum(dim=2)

        loss = self.loss_fn(prediction_logits / self.temperature, instance_targets.long())
        return loss

    def update(self, scaler: torch.cuda.amp.GradScaler) -> None:
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

    def __repr__(self,) -> str:
        repr = f"{self.__class__.__name__}(\n"
        repr = repr + f"    temperature={self.temperature},\n"
        repr = repr + f"    num_classes={self.num_classes},\n"
        repr = repr + f"    embedding_size={self.embedding_size},\n"
        repr = repr + f"    opt={self.opt.__class__.__name__},\n"
        repr = repr + f"    hierarchy_level={self.hierarchy_level},\n"
        repr = repr + ")"
        return repr
