from typing import Mapping, Callable, Union, List, Type
import os

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from omegaconf.dictconfig import DictConfig

import suprank.lib as lib
from suprank.losses.tools import XBM
from suprank.engine.autograd_all_gather import autograd_all_gather

NoneType = Type[None]


def _calculate_loss_and_backward(
    config: DictConfig,
    net: nn.Module,
    batch: Mapping[str, Tensor],
    relevance_fn: Callable,
    criterion: Callable,
    memory: Union[NoneType, XBM],
    optimizer: List[torch.optim.Optimizer],
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
) -> Mapping[str, float]:
    with torch.cuda.amp.autocast(enabled=config.model.net.with_autocast):
        di = net(batch["image"].cuda())
        labels = batch["label"].cuda()

        if memory is not None:
            mem_di, mem_labels = memory(di, labels)

        if config.distributed and config.split_batch:
            world_size = dist.get_world_size()
            ref_di = [torch.empty_like(di) for _ in range(world_size)]
            ref_lb = [torch.empty_like(labels) for _ in range(world_size)]
            autograd_all_gather(ref_di, di)
            dist.all_gather(ref_lb, labels)
            ref_di = torch.cat(ref_di).to('cuda', non_blocking=True)
            ref_lb = torch.cat(ref_lb).to('cuda', non_blocking=True)
        else:
            ref_di, ref_lb = di, labels

        logs = {}
        losses = []
        for crit, weight in criterion:
            loss = crit(
                di,
                labels,
                ref_embeddings=ref_di,
                ref_labels=ref_lb,
                relevance_fn=relevance_fn,
                indexes=batch["index"].cuda()
            )

            loss = loss.mean()

            if (memory is not None) and crit.takes_ref_embeddings and (epoch >= memory.activate):
                mem_loss = crit(
                    di,
                    labels,
                    ref_embeddings=mem_di,
                    ref_labels=mem_labels,
                    relevance_fn=relevance_fn,
                    indexes=batch["index"].cuda()
                )
                loss = (memory.factor_true_loss * loss + memory.factor * mem_loss) / (memory.factor_true_loss + memory.factor)

            losses.append(weight * loss)
            logs[f"{crit.__class__.__name__}_l{crit.hierarchy_level}"] = loss.item()

    total_loss = sum(losses)
    scaler.scale(total_loss).backward()
    for opt in optimizer.values():
        scaler.unscale_(opt)

    logs["total_loss"] = total_loss.item()
    _ = [loss.detach_() for loss in losses]
    total_loss.detach_()
    return logs


def base_training_loop(
    config: DictConfig,
    net: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: Callable,
    memory: Union[NoneType, XBM],
    optimizer: List[torch.optim.Optimizer],
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
) -> Mapping[str, float]:
    meter = lib.DictAverage()
    net.train()
    net.zero_grad()

    iterator = lib.track(loader)
    for i, batch in enumerate(iterator):
        logs = _calculate_loss_and_backward(
            config,
            net,
            batch,
            loader.dataset.compute_relevance_on_the_fly,
            criterion,
            memory,
            optimizer,
            scaler,
            epoch,
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

        for crit, _ in criterion:
            if hasattr(crit, 'update'):
                crit.update(scaler)

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

    return meter.avg
