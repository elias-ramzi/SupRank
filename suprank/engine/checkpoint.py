from typing import Callable, Mapping, Union, Type
import sys
from os.path import join

import torch
import torch.nn as nn
from omegaconf.dictconfig import DictConfig

import suprank.lib as lib
from suprank.losses.tools import XBM

NoneType = Type[None]


def checkpoint(
    log_dir: str,
    save_checkpoint: bool,
    net: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: Callable,
    memory: Union[NoneType, XBM],
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    config: DictConfig,
    metrics: Mapping[str, float],
) -> NoneType:
    state_dict = {}
    if config.distributed:
        state_dict["net_state"] = net.module.state_dict()
    else:
        state_dict["net_state"] = net.state_dict()

    state_dict["optimizer_state"] = {key: opt.state_dict() for key, opt in optimizer.items()}
    state_dict["scheduler"] = {key: sch.state_dict() for key, sch in scheduler.items()}
    state_dict["criterion_state"] = [crit.state_dict() for crit, _ in criterion]
    state_dict["scaler_state"] = scaler.state_dict()

    state_dict["epoch"] = epoch
    state_dict["config"] = config
    state_dict["command"] = 'python ' + ' '.join(sys.argv)
    state_dict["metrics"] = metrics

    RANDOM_STATE = lib.get_random_state()
    state_dict.update(RANDOM_STATE)

    torch.save(state_dict, join(log_dir, 'weights', "rolling.ckpt"))
    if save_checkpoint:
        lib.LOGGER.info(f"Checkpoint of epoch {epoch} created")
        torch.save(state_dict, join(log_dir, 'weights', f"epoch_{epoch}.ckpt"))
