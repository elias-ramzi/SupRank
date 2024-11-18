from typing import Optional, Union

import torch
from torch import Tensor


def create_label_matrix(
    labels: Tensor,
    other_labels: Optional[Tensor] = None,
    hierarchy_level: Optional[Union[int, str]] = None,
    dtype: torch.dtype = torch.float,
) -> Tensor:
    if other_labels is None:
        other_labels = labels

    if (hierarchy_level is not None) and (hierarchy_level != "MULTI"):
        labels = labels[..., hierarchy_level:hierarchy_level + 1]
        other_labels = other_labels[..., hierarchy_level:hierarchy_level + 1]

    return (labels.unsqueeze(1) == other_labels.unsqueeze(0)).sum(-1).squeeze().to(dtype)
