from typing import Optional, List, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import suprank.lib as lib
from suprank.models.get_backbone import get_backbone
from suprank.models.get_pooling import get_pooling
from suprank.models.layers import GeM

NoneType = Type[None]


class Normalize(nn.Module):

    def __init__(self, test_time_only: bool = False) -> NoneType:
        super().__init__()
        self.test_time_only = test_time_only

    def forward(self, X: Tensor) -> Tensor:
        if self.training and self.test_time_only:
            return X

        dtype = X.dtype
        return F.normalize(X, p=2, dim=-1).to(dtype)


class RetrievalNet(nn.Module):

    def __init__(
        self,
        backbone_name: str,
        embed_dim: int = 512,
        normalize: bool = True,
        norm_features: bool = False,
        without_fc: bool = False,
        with_autocast: bool = True,
        pooling: str = 'default',
        pretrained: bool = True,
        whiten: Optional[str] = None,
    ) -> NoneType:
        super().__init__()
        norm_features = lib.str_to_bool(norm_features)
        without_fc = lib.str_to_bool(without_fc)
        with_autocast = lib.str_to_bool(with_autocast)

        self.embed_dim = embed_dim
        self.normalize = normalize
        self.norm_features = norm_features
        self.without_fc = without_fc
        self.with_autocast = with_autocast
        self.whiten = whiten
        if with_autocast:
            lib.LOGGER.info("Using mixed precision")

        self.backbone, default_pooling, out_features = get_backbone(backbone_name, pretrained=pretrained)
        self.pooling = get_pooling(default_pooling, pooling)
        lib.LOGGER.info(f"Pooling is {self.pooling}")
        self.embed_dim = self.embed_dim or out_features

        self.standardize = nn.LayerNorm(out_features, elementwise_affine=False) if self.norm_features else nn.Identity()
        self.standardize = Normalize() if (self.whiten is not None) else self.standardize
        lib.LOGGER.info(f"{'U' if self.norm_features else 'Not u'}sing a LayerNorm layer")

        self.fc = nn.Identity() if self.without_fc else nn.Linear(out_features, embed_dim)
        lib.LOGGER.info("Not using a linear projection layer") if self.without_fc else lib.LOGGER.info(f"Projection head: {self.fc}")

        if self.whiten is not None:
            state_dict = torch.load(lib.expand_path(self.whiten), map_location='cpu')
            self.fc.load_state_dict(state_dict)

        self.head = nn.Sequential(
            self.pooling,
            nn.Flatten(),
            self.standardize,
            self.fc,
            Normalize(test_time_only=not self.normalize),
        )

    @torch.no_grad()
    def forward_multiscale(self, X: Tensor, scales: List[float]) -> Tensor:
        """
        adapted from:
        https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/1e66a417afa2247edde6d35f3a9a2a465778a3a8/cirtorch/networks/imageretrievalnet.py#L309
        """
        power = 1.0 if not isinstance(self.pooling, GeM) else self.pooling.p.item()

        embedding = torch.zeros(len(X), self.embed_dim, device=X.device)

        for s in scales:
            input_t = F.interpolate(X, scale_factor=s, mode='bilinear', align_corners=True)
            embedding += self.fc(self.standardize(nn.Flatten()(self.pooling(self.backbone(input_t))))).pow(power)  # we do not normalize here

        embedding /= len(scales)
        embedding = embedding.pow(1. / power)
        return F.normalize(embedding)

    def forward(self, X: Tensor, return_before_fc: bool = False, scales: Optional[List[float]] = None) -> Tensor:
        if scales is not None:
            return self.forward_multiscale(X, scales=scales)

        with torch.cuda.amp.autocast(enabled=self.with_autocast or (not self.training)):
            X = self.backbone(X)

            if return_before_fc:
                X = self.pooling(X)
                X = nn.Flatten()(X)
                X = self.standardize(X)
                return X

            X = self.head(X)
            return X
