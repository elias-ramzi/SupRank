from typing import Tuple, Mapping, Union, List, Callable, Type

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, Sampler

from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

import suprank.lib as lib
from suprank.engine import AccuracyCalculator

NoneType = Type[None]


class Getter:
    """
    This class allows to create differents object (model, loss functions, optimizer...)
    based on the config
    """
    def __init__(self, config: DictConfig) -> NoneType:
        self.config = config

    def get_optimizer(self, net: nn.Module) -> Tuple[Mapping[str, Union[Optimizer, _LRScheduler]]]:
        optimizers = instantiate(self.config.optimizer.opt)
        schedulers = instantiate(self.config.optimizer.sch)
        for key in optimizers.keys():
            if key == 'net':
                params = net.parameters()
            else:
                params = getattr(net, key).parameters()
            optimizers[key] = optimizers[key](params)
            schedulers[key] = schedulers[key](optimizers[key])

        lib.LOGGER.info(optimizers)
        lib.LOGGER.info(schedulers)
        return optimizers, schedulers

    def get_loss(self) -> List[Tuple[Callable, float]]:
        criterion = instantiate(self.config.loss.losses)
        weights = self.config.loss.weight
        if weights is None:
            weights = [1 / len(criterion) for _ in range(len(criterion))]
        assert len(weights) == len(criterion)

        for i in range(len(criterion)):
            lib.LOGGER.info(f"{criterion[i]} with weight {weights[i]}")

        return [(loss, w) for loss, w in zip(criterion, weights)]

    def get_sampler(self, dataset: Dataset) -> Sampler:
        kwargs = {}
        if self.config.distributed:
            kwargs['split_batch'] = self.config.split_batch
        sampler = instantiate(self.config.dataset.sampler, dataset, **kwargs)
        lib.LOGGER.info(sampler)
        return sampler

    def get_dataset(self) -> Dataset:
        dataset = instantiate(self.config.dataset.dts)
        lib.LOGGER.info(dataset)
        return dataset

    def get_model(self) -> nn.Module:
        net = instantiate(self.config.model.net)
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net) if (self.config.distributed and self.config.model.sync_bn) else net
        if self.config.model.freeze_batch_norm:
            lib.LOGGER.info("Freezing batch norm")
            net = lib.freeze_batch_norm(net)
        else:
            lib.LOGGER.info("/!\\ Not freezing batch norm")
        return net

    def get_memory(self) -> nn.Module:
        memory = instantiate(self.config.memory)
        lib.LOGGER.info(memory)
        if not memory:
            return None
        return memory

    def get_acc_calculator(self) -> AccuracyCalculator:
        acc = instantiate(self.config.evaluation)
        lib.LOGGER.info(acc)
        return acc
