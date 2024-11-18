from typing import Tuple, Callable, Mapping, Union, List, Type
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, DataLoader
from torch import Tensor
from omegaconf.dictconfig import DictConfig

import suprank.lib as lib
from suprank.losses.tools import XBM

NoneType = Type[None]


@torch.no_grad()
def _compute_descriptors(net: nn.Module, dataset: Dataset, config: DictConfig) -> Tuple[Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=config.sub_batch,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )

    descriptors = torch.empty(len(dataset), net.embed_dim, dtype=torch.float, device='cuda')
    net.requires_grad_(False)
    for batch in loader:
        # computes descriptors without keeping activations for backpropagations
        with torch.cuda.amp.autocast(enabled=config.model.net.with_autocast):
            X = net(batch["image"].to('cuda', non_blocking=True))
        descriptors[batch["index"]] = X.float()

    labels = torch.from_numpy(dataset.labels)[:, 0:1].to('cuda', non_blocking=True)
    net.requires_grad_(True)
    return descriptors, labels


def run_large_batch(
    net: nn.Module,
    dataset: Dataset,
    criterion: Callable,
    scaler: torch.cuda.amp.GradScaler,
    optimizer: List[torch.optim.Optimizer],
    relevance_fn: Callable,
    config: DictConfig,
) -> Mapping[str, float]:
    """
    Memory effective backpropagation algorithm for optimizing a
    loss function, as described in the paper
    https://arxiv.org/abs/1906.07589
    """
    descriptors, labels = _compute_descriptors(net, dataset, config)

    descriptors.requires_grad_(True)
    descriptors.retain_grad()

    with torch.cuda.amp.autocast(enabled=config.model.net.with_autocast):
        losses = []
        logs = {}
        for crit, weight in criterion:
            loss = crit(
                descriptors,
                labels,
                relevance_fn=relevance_fn,
            )

            losses.append((weight * loss))
            logs[f"{crit.__class__.__name__}_l{crit.hierarchy_level}"] = loss.item()

    total_loss = sum(losses)
    scaler.scale(total_loss).backward()
    for opt in optimizer.values():
        scaler.unscale_(opt)

    logs["total_loss"] = total_loss.item()

    grad = descriptors.grad
    descriptors.grad = None

    loader = DataLoader(
        dataset,
        batch_size=config.sub_batch,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=True,
    )

    logs = {}
    for batch in loader:
        with torch.cuda.amp.autocast(enabled=config.model.net.with_autocast):
            di = net(batch["image"].to('cuda', non_blocking=True))

        di.backward(grad[batch["index"]].to('cuda', non_blocking=True))

    return logs


def large_batch_training_loop(
    config: DictConfig,
    net: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: Callable,
    memory: Union[NoneType, XBM],
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
) -> Mapping[str, float]:
    meter = lib.DictAverage()
    net.train()
    net.zero_grad()

    iterator = lib.track(loader.sampler.batches)
    for i, batch in enumerate(iterator):

        sub_dataset = Subset(loader.dataset, batch)

        logs = run_large_batch(
            net,
            sub_dataset,
            criterion,
            scaler,
            loader.dataset.compute_relevance_on_the_fly,
            config,
        )

        if config.record_gradient:
            logs["gradient_norm"] = lib.get_gradient_norm(net)

        if config.gradient_clipping_norm is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.gradient_clipping_norm)

        for key, opt in optimizer.items():
            if (config.warmup_step is not None) and (config.warmup_step >= epoch) and (key in config.warmup_keys):
                if i == 0:
                    lib.LOGGER.warning("Warmimg UP")
                continue
            scaler.step(opt)

        net.zero_grad()
        _ = [crit.zero_grad() for crit, w in criterion]
        scaler.update()

        meter.update(logs)
        if not os.getenv('TQDM_DISABLE'):
            iterator.set_postfix(meter.avg)
        else:
            if (i + 1) % config.print_freq == 0:
                lib.LOGGER.info(f'Iteration : {i}/{len(loader)}')
                for k, v in logs.items():
                    lib.LOGGER.info(f'Loss: {k}: {v} ')

        if config.DEBUG:
            if (i + 1) > int(config.DEBUG):
                break

    return logs
