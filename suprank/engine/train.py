from typing import Callable, Mapping, Union, Type
from time import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, Sampler, DataLoader
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

import suprank.lib as lib
from suprank.losses.tools import XBM
from suprank.engine.checkpoint import checkpoint
from suprank.engine.base_training_loop import base_training_loop
from suprank.engine.accuracy_calculator import evaluate, AccuracyCalculator

NoneType = Type[None]


def train(
    config: DictConfig,
    log_dir: str,
    is_master: bool,
    net: nn.Module,
    criterion: Callable,
    memory: Union[NoneType, XBM],
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    acc: AccuracyCalculator,
    train_dts: Dataset,
    test_dts: Dataset,
    sampler: Sampler,
    writer: SummaryWriter,
    restore_epoch: int,
) -> Mapping[str, float]:
    # """""""""""""""""" Iter over epochs """"""""""""""""""""""""""
    lib.LOGGER.info(f"Training of model {config.experiment_name}")

    metrics = None
    for e in range(1 + restore_epoch, config.max_iter + 1):

        lib.LOGGER.info(f"Training : @epoch #{e} for model {config.experiment_name}")
        start_time = time()

        # """""""""""""""""" Training Loop """"""""""""""""""""""""""
        sampler.set_epoch(e)
        loader = DataLoader(
            train_dts,
            batch_sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
        logs = base_training_loop(
            config=config,
            net=net,
            loader=loader,
            criterion=criterion,
            optimizer=optimizer,
            memory=memory,
            scaler=scaler,
            epoch=e,
        )

        if (config.warmup_step is not None) and (config.warmup_step >= e):
            pass
        else:
            for sch in scheduler.values():
                sch.step()

            for crit, _ in criterion:
                if hasattr(crit, 'on_epoch'):
                    crit.on_epoch()

        end_train_time = time()

        if not is_master:
            continue

        dataset_dict = {}
        if (config.train_eval_freq > -1) and ((e % config.train_eval_freq == 0) or (e == config.max_iter)):
            dataset_dict["train"] = train_dts

        if (config.test_eval_freq > -1) and ((e % config.test_eval_freq == 0) or (e == config.max_iter)):
            if isinstance(test_dts, (ListConfig, list)):
                for i, _dts in enumerate(test_dts):
                    dataset_dict[f"test_level{i}"] = _dts
            else:
                dataset_dict["test"] = test_dts

        metrics = None
        if dataset_dict:
            metrics = evaluate(
                net=net,
                dataset_dict=dataset_dict,
                acc=acc,
                epoch=e,
            )
            torch.cuda.empty_cache()

        # """""""""""""""""" Logging Step """"""""""""""""""""""""""
        for grp, opt in optimizer.items():
            writer.add_scalar(f"LR/{grp}", list(lib.get_lr(opt).values())[0], e)

        for k, v in logs.items():
            lib.LOGGER.info(f"{k} : {v:.4f}")
            writer.add_scalar(f"SupRank/Train/{k}", v, e)

        if metrics is not None:
            for split, mtrc in metrics.items():
                for k, v in mtrc.items():
                    if k == 'epoch':
                        continue
                    lib.LOGGER.info(f"{split} --> {k} : {np.around(v*100, decimals=2)}")
                    writer.add_scalar(f"SupRank/{split.title()}/Evaluation/{k}", v, e)
                print()

        end_time = time()

        elapsed_time = lib.format_time(end_time - start_time)
        elapsed_time_train = lib.format_time(end_train_time - start_time)
        elapsed_time_eval = lib.format_time(end_time - end_train_time)

        lib.LOGGER.info(f"Epoch took : {elapsed_time}")
        if metrics is not None:
            lib.LOGGER.info(f"Training loop took : {elapsed_time_train}")
            lib.LOGGER.info(f"Evaluation step took : {elapsed_time_eval}")

        print()
        print()

        # """""""""""""""""" Checkpointing """"""""""""""""""""""""""
        checkpoint(
            log_dir=log_dir,
            save_checkpoint=(e % config.save_model == 0) or (e == config.max_iter),
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            memory=memory,
            scaler=scaler,
            epoch=e,
            config=config,
            metrics=metrics,
        )

    return metrics
