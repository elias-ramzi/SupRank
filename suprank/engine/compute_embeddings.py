from typing import Mapping, Any, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

import suprank.lib as lib

KwargsType = Mapping[str, Any]


def compute_embeddings(
    net: nn.Module,
    loader: DataLoader,
    convert_to_cuda: bool = False,
    with_paths: bool = False,
    **kwargs: KwargsType,
) -> Tuple[Tensor]:
    features = []

    mode = net.training
    net.eval()
    lib.LOGGER.info("Computing embeddings")
    for i, batch in enumerate(lib.track(loader, "Computing embeddings")):
        with torch.no_grad():
            X = net(batch["image"].cuda(), **kwargs)

        features.append(X)

    features = torch.cat(features)
    labels = torch.from_numpy(loader.dataset.labels).to('cuda' if convert_to_cuda else 'cpu')
    if loader.dataset.relevances is not None:
        relevances = loader.dataset.relevances.to('cuda' if convert_to_cuda else 'cpu')
    else:
        relevances = None

    net.train(mode)
    if with_paths:
        return features, labels, relevances, loader.dataset.paths
    else:
        return features, labels, relevances
