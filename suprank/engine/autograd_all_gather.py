from typing import List, Tuple, Type

import torch
import torch.distributed as dist
from torch import Tensor

NoneType = Type[None]


# from :
# https://github.com/vlkit/vlkit/blob/master/vlkit/ops/distributed.py#L4-L25
class AllGather(torch.autograd.Function):
    """
    all_gather with gradient back-propagation
    """
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, tensor_list: List[Tensor], tensor: Tensor) -> Tuple[Tensor]:  # noqa: ANN001
        dist.all_gather(tensor_list, tensor)
        return tuple(tensor_list)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, *grad_list: List[Tensor]) -> Tuple[NoneType, Tensor]:  # noqa: ANN001
        grad_list = list(grad_list)
        rank = dist.get_rank()

        dist_ops = [
            dist.reduce(grad_list[i], i, async_op=True) for i in range(dist.get_world_size())
        ]

        for op in dist_ops:
            op.wait()

        return None, grad_list[rank]


autograd_all_gather = AllGather.apply


if __name__ == '__main__':

    from suprank.losses.ranking_loss import SmoothAPLoss

    crit = SmoothAPLoss(hierarchy_level=0)

    number_positives = 4
    batch_size = 256
    assert batch_size % number_positives == 0

    # DDP
    repeats = 4

    torch.manual_seed(0)
    labels = torch.arange(batch_size // number_positives).repeat_interleave(number_positives)
    labels = labels[torch.randperm(len(labels))].view(-1, 1)
    embeddings = torch.nn.functional.normalize(torch.randn(len(labels), 384))
    embeddings.requires_grad_(True)

    loss = crit(embeddings, labels, relevance_fn=None)
    loss.backward()
    grad = torch.clone(embeddings.grad)

    embeddings.grad = None

    queries = []
    for rank in range(len(repeats)):
        queries.append(embeddings[rank * (batch_size // repeats): (rank + 1) * (batch_size // repeats)])
