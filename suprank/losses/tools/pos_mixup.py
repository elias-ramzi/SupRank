from typing import Tuple

import numpy as np
import torch
from torch import Tensor


def _pos_mixup(tensor: Tensor, num_id: int) -> Tensor:
    batch_size = tensor.shape[0]
    num_pos = int(batch_size / num_id)
    for i in range(0, batch_size, num_pos):
        if num_pos == 6:
            alpha = np.random.rand()
            fake_1 = alpha * tensor[i, :] + (1. - alpha) * tensor[i + 1, :]
            fake_1 = torch.unsqueeze(fake_1, 0)

            alpha = np.random.rand()
            fake_2 = alpha * tensor[i + 1, :] + (1. - alpha) * tensor[i + 2, :]
            fake_2 = torch.unsqueeze(fake_2, 0)

            alpha = np.random.rand()
            fake_3 = alpha * tensor[i + 2, :] + (1. - alpha) * tensor[i + 3, :]
            fake_3 = torch.unsqueeze(fake_3, 0)

            alpha = np.random.rand()
            fake_4 = alpha * tensor[i + 3, :] + (1. - alpha) * tensor[i + 4, :]
            fake_4 = torch.unsqueeze(fake_4, 0)

            alpha = np.random.rand()
            fake_5 = alpha * tensor[i + 4, :] + (1. - alpha) * tensor[i + 5, :]
            fake_5 = torch.unsqueeze(fake_5, 0)

            alpha = np.random.rand()
            fake_6 = alpha * tensor[i, :] + (1. - alpha) * tensor[i + 2, :]
            fake_6 = torch.unsqueeze(fake_6, 0)

            alpha = np.random.rand()
            fake_7 = alpha * tensor[i, :] + (1. - alpha) * tensor[i + 3, :]
            fake_7 = torch.unsqueeze(fake_7, 0)

            alpha = np.random.rand()
            fake_8 = alpha * tensor[i, :] + (1. - alpha) * tensor[i + 4, :]
            fake_8 = torch.unsqueeze(fake_8, 0)

            alpha = np.random.rand()
            fake_9 = alpha * tensor[i, :] + (1. - alpha) * tensor[i + 5, :]
            fake_9 = torch.unsqueeze(fake_9, 0)

            alpha = np.random.rand()
            fake_10 = alpha * tensor[i + 1, :] + (1. - alpha) * tensor[i + 3, :]
            fake_10 = torch.unsqueeze(fake_10, 0)

            alpha = np.random.rand()
            fake_11 = alpha * tensor[i + 1, :] + (1. - alpha) * tensor[i + 4, :]
            fake_11 = torch.unsqueeze(fake_11, 0)

            alpha = np.random.rand()
            fake_12 = alpha * tensor[i + 1, :] + (1. - alpha) * tensor[i + 5, :]
            fake_12 = torch.unsqueeze(fake_12, 0)

            alpha = np.random.rand()
            fake_13 = alpha * tensor[i + 2, :] + (1. - alpha) * tensor[i + 4, :]
            fake_13 = torch.unsqueeze(fake_13, 0)

            alpha = np.random.rand()
            fake_14 = alpha * tensor[i + 2, :] + (1. - alpha) * tensor[i + 5, :]
            fake_14 = torch.unsqueeze(fake_14, 0)

            alpha = np.random.rand()
            fake_15 = alpha * tensor[i + 3, :] + (1. - alpha) * tensor[i + 5, :]
            fake_15 = torch.unsqueeze(fake_15, 0)

            if i == 0:
                tensor_fake = torch.cat((fake_1, fake_2, fake_3, fake_4, fake_5, fake_6, fake_7, fake_8, fake_9, fake_10, fake_11, fake_12, fake_13, fake_14, fake_15), dim=0)
            else:
                tensor_fake = torch.cat((tensor_fake, fake_1, fake_2, fake_3, fake_4, fake_5, fake_6, fake_7, fake_8, fake_9, fake_10, fake_11, fake_12, fake_13, fake_14, fake_15), dim=0)

        if num_pos == 4:
            alpha = np.random.rand()
            fake_1 = alpha * tensor[i, :] + (1. - alpha) * tensor[i + 1, :]
            fake_1 = torch.unsqueeze(fake_1, 0)

            alpha = np.random.rand()
            fake_2 = alpha * tensor[i + 1, :] + (1. - alpha) * tensor[i + 2, :]
            fake_2 = torch.unsqueeze(fake_2, 0)

            alpha = np.random.rand()
            fake_3 = alpha * tensor[i + 2, :] + (1. - alpha) * tensor[i + 3, :]
            fake_3 = torch.unsqueeze(fake_3, 0)

            alpha = np.random.rand()
            fake_4 = alpha * tensor[i, :] + (1. - alpha) * tensor[i + 3, :]
            fake_4 = torch.unsqueeze(fake_4, 0)

            alpha = np.random.rand()
            fake_5 = alpha * tensor[i, :] + (1. - alpha) * tensor[i + 2, :]
            fake_5 = torch.unsqueeze(fake_5, 0)

            alpha = np.random.rand()
            fake_6 = alpha * tensor[i + 1, :] + (1. - alpha) * tensor[i + 3, :]
            fake_6 = torch.unsqueeze(fake_6, 0)

            if i == 0:
                tensor_fake = torch.cat((fake_1, fake_2, fake_3, fake_4, fake_5, fake_6), dim=0)
            else:
                tensor_fake = torch.cat((tensor_fake, fake_1, fake_2, fake_3, fake_4, fake_5, fake_6), dim=0)

        elif num_pos == 3:
            alpha = np.random.rand()
            fake_1 = alpha * tensor[i, :] + (1. - alpha) * tensor[i + 1, :]
            fake_1 = torch.unsqueeze(fake_1, 0)

            alpha = np.random.rand()
            fake_2 = alpha * tensor[i + 1, :] + (1. - alpha) * tensor[i + 2, :]
            fake_2 = torch.unsqueeze(fake_2, 0)

            alpha = np.random.rand()
            fake_3 = alpha * tensor[i, :] + (1. - alpha) * tensor[i + 2, :]
            fake_3 = torch.unsqueeze(fake_3, 0)

            if i == 0:
                tensor_fake = torch.cat((fake_1, fake_2, fake_3), dim=0)
            else:
                tensor_fake = torch.cat((tensor_fake, fake_1, fake_2, fake_3), dim=0)

    ind = num_pos
    if num_pos == 6:
        num_fakes = 15
    elif num_pos == 4:
        num_fakes = 6
    elif num_pos == 3:
        num_fakes = 3

    for i in range(0, tensor_fake.shape[0], num_fakes):
        tensor = torch.cat((tensor[:ind, :], tensor_fake[i:i + num_fakes, :], tensor[ind:, :]), dim=0)
        ind += num_pos + num_fakes

    return tensor


def _pos_mixup_binary_quick(scores: Tensor, target: Tensor, force_mixup: bool = False) -> Tuple[Tensor]:
    num_pos = target.sum(-1).min().item()
    if force_mixup:
        target = torch.clone(target)
        max_num_pos = target.sum(-1).max().item()
        target.scatter_(1, target.argsort(-1, descending=True)[:, num_pos:max_num_pos], 0)
    number_of_combination = int((num_pos * (num_pos - 1)) / 2)
    positive_samples = torch.where(target)[1].view(len(target), num_pos)  # get indices of positive sample for each query-

    indices_tuple = torch.empty(len(positive_samples), number_of_combination, 2, dtype=torch.long, device=scores.device)
    for i in range(len(positive_samples)):
        comb = torch.combinations(positive_samples[i], r=2)  # for each query, get all positive combination possible
        indices_tuple[i] = comb

    alpha = torch.rand(len(scores), number_of_combination, device=scores.device, dtype=scores.dtype)  # get a different alpha for each combination
    fake_scores = alpha * scores.gather(1, indices_tuple[..., 0]) + (1 - alpha) * scores.gather(1, indices_tuple[..., 1])  # mix each combination
    fake_target = torch.ones_like(fake_scores)  # create the fake targets

    return fake_scores, fake_target


def pos_mixup(scores: Tensor, target: Tensor, force_mixup: bool = False) -> Tuple[Tensor]:
    if (target > 1).any():
        raise NotImplementedError("pos_mixup not implemented for hierarchical retrieval")

    num_pos = target.sum(-1)
    if (not (len(num_pos.unique()) == 1)) and (not force_mixup):  # same number of positives for each query
        raise NotImplementedError("pos_mixup not implemented for queries with different number of positives")
    fake_scores, fake_target = _pos_mixup_binary_quick(scores, target, force_mixup=force_mixup)

    mix_scores = torch.cat((scores, fake_scores), dim=1)
    mix_target = torch.cat((target, fake_target), dim=1)

    return mix_scores, mix_target


if __name__ == '__main__':

    number_positives = 4
    batch_size = 32
    assert batch_size % number_positives == 0

    torch.manual_seed(0)
    labels = torch.arange(8).repeat_interleave(number_positives)
    labels = labels[torch.randperm(len(labels))]
    target = labels.view(-1, 1) == labels.view(1, -1)
    # target.fill_diagonal_(False)
    scores = torch.nn.functional.normalize(torch.randn(len(labels), len(labels)))
    print(len(labels))
    print(labels)

    fake_scores, fake_target = _pos_mixup_binary_quick(scores, target)
    print(fake_scores.shape)

    mix_scores, mix_target = pos_mixup(scores, target)
    print(mix_scores.shape)
