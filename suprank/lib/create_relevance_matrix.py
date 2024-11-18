import torch
from torch import Tensor


def create_relevance_matrix(label_matrix: Tensor, relevance: Tensor) -> Tensor:
    return torch.gather(relevance, 1, label_matrix)
