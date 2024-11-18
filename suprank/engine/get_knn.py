from typing import Union, Tuple

import torch
from torch import Tensor

import suprank.lib as lib


def get_knn(queries: Tensor, references: Tensor, num_k: int, embeddings_come_from_same_source: Union[int, bool]) -> Tuple[Tensor]:
    lib.LOGGER.debug("Computing k-nn with torch")
    num_k += int(embeddings_come_from_same_source)

    scores = queries @ references.t()
    distances, indices = torch.topk(scores, num_k)

    if embeddings_come_from_same_source:
        return indices[:, 1:], distances[:, 1:]

    return indices, distances
