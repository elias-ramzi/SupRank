from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
from torch import Tensor

NoneType = Type[None]


def pca(
    features: Tensor,
    n_principal_components: Optional[int] = None,
    on_cpu: bool = False,
) -> Tuple[Tensor]:
    if n_principal_components is None:
        n_principal_components = features.size(1)

    assert n_principal_components <= features.size(1)

    if on_cpu:
        features = features.cpu()

    mean = features.mean(dim=0)
    n_samples = features.size(0)
    X = features - mean
    U, S, V = torch.linalg.svd(X.float(), full_matrices=False)
    max_abs_cols = U.abs().argmax(0)
    signs = torch.sign(U[max_abs_cols, range(U.size(1))])
    U *= signs
    V *= signs.unsqueeze(1)

    components_ = V[:n_principal_components]
    explained_variance_ = ((S ** 2) / (n_samples - 1))[:n_principal_components]

    return mean, components_, explained_variance_


def create_pca_layer(mean: Tensor, components_: Tensor, explained_variance_: Tensor, whiten: bool = True) -> nn.Module:

    weight = components_
    bias = - mean @ components_.T

    if whiten:
        exvar = torch.sqrt(explained_variance_)
        weight /= exvar.unsqueeze(1)
        bias /= exvar

    pca_layer = nn.Linear(components_.size(1), components_.size(0))
    pca_layer.weight.data = weight.cpu()
    pca_layer.bias.data = bias.cpu()

    return pca_layer
