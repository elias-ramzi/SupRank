# MIT License
#
# Copyright (c) 2019 Autonomous Learning Group
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from typing import Mapping, Any, Tuple, Optional, Type

import torch
from torch import Tensor

import suprank.lib as lib

NoneType = Type[None]
KwargsType = Mapping[str, Any]


def rank(seq: Tensor) -> Tensor:
    return torch.argsort(torch.argsort(seq).flip(1))


def rank_normalised(seq: Tensor) -> Tensor:
    return (rank(seq) + 1).float() / seq.size()[1]


class TrueRanker(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, sequence: Tensor, lambda_val: float) -> Tensor:  # noqa: ANN001
        rank = rank_normalised(sequence)
        ctx.lambda_val = lambda_val
        ctx.save_for_backward(sequence, rank)
        return rank

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, NoneType]:  # noqa: ANN001
        sequence, rank = ctx.saved_tensors
        assert grad_output.shape == rank.shape
        sequence_prime = sequence + ctx.lambda_val * grad_output
        rank_prime = rank_normalised(sequence_prime)
        gradient = -(rank - rank_prime) / (ctx.lambda_val + 1e-8)
        return gradient, None


class BlackBoxAPLoss(torch.nn.Module):
    """ Torch module for computing recall-based loss as in 'Blackbox differentiation of Ranking-based Metrics' """

    takes_ref_embeddings: bool = True

    def __init__(
        self,
        lambda_val: float = 4.,
        margin: float = 0.02,
        hierarchy_level: int = 0,
        return_type: str = '1-mAP',
    ) -> NoneType:
        """
        :param lambda_val:  hyperparameter of black-box backprop
        :param margin: margin to be enforced between positives and negatives (alpha in the paper)
        :param interclass_coef: coefficient for interclass loss (beta in paper)
        :param batch_memory: how many batches should be in memory
        """
        super().__init__()
        assert return_type in ["AP", "mAP", "1-mAP", "1-AP"]
        self.lambda_val = lambda_val
        self.margin = margin
        self.hierarchy_level = hierarchy_level
        self.return_type = return_type

    def raw_map_computation(self, scores: Tensor, targets: Tensor) -> Tensor:
        """
                :param scores: NxM predicted similarity scores
                :param targets: NxM ground truth relevances
        """
        # Compute map
        HIGH_CONSTANT = 2.0
        epsilon = 1e-5
        deviations = torch.abs(torch.randn_like(targets, device=scores.device, dtype=scores.dtype)) * (targets - 0.5)

        scores = scores - self.margin * deviations
        ranks_of_positive = TrueRanker.apply(scores, self.lambda_val)
        scores_for_ranking_positives = -ranks_of_positive + HIGH_CONSTANT * targets
        ranks_within_positive = rank_normalised(scores_for_ranking_positives)
        ranks_within_positive.requires_grad = False
        assert torch.all(ranks_within_positive * targets < ranks_of_positive * targets + epsilon)

        sum_of_precisions_at_j_per_class = ((ranks_within_positive / ranks_of_positive) * targets).sum(dim=1)
        precisions_per_class = sum_of_precisions_at_j_per_class / (targets.sum(dim=1) + epsilon)

        if self.return_type == '1-mAP':
            return 1.0 - precisions_per_class.mean()
        elif self.return_type == '1-AP':
            return 1.0 - precisions_per_class
        elif self.return_type == 'mAP':
            return precisions_per_class.mean()
        elif self.return_type == 'AP':
            return precisions_per_class

    def forward(
        self,
        embeddings: Tensor,
        labels: Tensor,
        ref_embeddings: Optional[Tensor] = None,
        ref_labels: Optional[Tensor] = None,
        **kwargs: KwargsType,
    ) -> Tensor:
        ref_embeddings, ref_labels = ref_embeddings if ref_embeddings is not None else embeddings, ref_labels if ref_labels is not None else labels

        scores = embeddings @ ref_embeddings.t()
        target = lib.create_label_matrix(labels, ref_labels, self.hierarchy_level)

        return self.raw_map_computation(scores, target.type(scores.dtype))

    def extra_repr(self,) -> str:
        return f"lambda_val={self.lambda_val}, margin={self.margin}, return_type={self.return_type}"
