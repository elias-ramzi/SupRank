from typing import Tuple, Mapping, Any
import os
from os.path import join

import torch
from torch.utils.tensorboard import SummaryWriter
from omegaconf.dictconfig import DictConfig

from suprank.lib.logger import LOGGER
from suprank.lib.expand_path import expand_path


def set_experiment(config: DictConfig, is_master: bool) -> Tuple[DictConfig, str, SummaryWriter, Mapping[str, Any], int, bool]:
    # """""""""""""""""" Handle Config """"""""""""""""""""""""""
    config.log_dir = expand_path(config.log_dir)
    log_dir = join(config.log_dir, config.experiment_name)
    stop_process = False

    if 'debug' in config.experiment_name.lower():
        config.DEBUG = config.DEBUG or 1

    if config.resume is not None:
        if os.path.isfile(expand_path(config.resume)):
            resume = expand_path(config.resume)
        else:
            resume = os.path.join(log_dir, 'weights', config.resume)
            if not os.path.isfile(resume):
                LOGGER.warning("Checkpoint does not exists")
                stop_process = True

        state = torch.load(resume, map_location='cpu')
        at_epoch = state["epoch"]
        if at_epoch >= config.max_iter:
            LOGGER.warning(f"Exiting trial, experiment {config.experiment_name} already finished")
            stop_process = True

        LOGGER.info(f"Resuming from state : {resume}")
        restore_epoch = state['epoch']

    else:
        resume = None
        state = None
        restore_epoch = 0
        if os.path.isdir(os.path.join(log_dir, 'weights')) and not config.DEBUG:
            if is_master:
                LOGGER.warning(f"Exiting trial, experiment {config.experiment_name} already exists")
                LOGGER.warning(f"Its access: {log_dir}")
            stop_process = True

    writer = None
    if is_master:
        os.makedirs(join(log_dir, 'logs'), exist_ok=True)
        os.makedirs(join(log_dir, 'weights'), exist_ok=True)
        writer = SummaryWriter(join(log_dir, "logs"), purge_step=restore_epoch)

    # """""""""""""""""" Create Data """"""""""""""""""""""""""
    os.environ['USE_CUDA_FOR_RELEVANCE'] = 'yes'

    if config.distributed:
        _stop_process = [stop_process]
        torch.distributed.broadcast_object_list(_stop_process)
        stop_process = _stop_process[0]

    return config, log_dir, writer, state, restore_epoch, stop_process
