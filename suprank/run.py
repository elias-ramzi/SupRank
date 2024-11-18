from typing import Union, Any, Mapping
import numpy as np
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import hydra
import torch

import suprank.lib as lib
import suprank.engine as eng
from suprank.getter import Getter


def if_func(cond: Union[str, bool], x: Any, y: Any) -> Any:  # noqa ANN401
    if not isinstance(cond, bool):
        cond = eval(cond)
        assert isinstance(cond, bool)
    if cond:
        return x
    return y


OmegaConf.register_new_resolver("mult", lambda *numbers: float(np.prod([float(x) for x in numbers])))
OmegaConf.register_new_resolver("sum", lambda *numbers: sum(map(float, numbers)))
OmegaConf.register_new_resolver("sub", lambda x, y: float(x) - float(y))
OmegaConf.register_new_resolver("div", lambda x, y: float(x) / float(y))
OmegaConf.register_new_resolver("if", if_func)


@hydra.main(config_path='config', config_name='default', version_base="1.1")
def run(config: DictConfig) -> Mapping[str, float]:
    """
    creates all objects required to launch a training
    """
    is_master, local_rank = lib.set_distributed(distributed=config.distributed, is_cluster=config.is_cluster)
    config, log_dir, writer, state, restore_epoch, stop_process = lib.set_experiment(config, is_master)
    if stop_process:
        return

    getter = Getter(config)

    # """""""""""""""""" Handle Reproducibility"""""""""""""""""""""""""
    lib.random_seed(config.seed)

    # """""""""""""""""" Create Data """"""""""""""""""""""""""
    datasets = getter.get_dataset()
    train_dts, test_dts = datasets['train'], datasets['test']
    sampler = getter.get_sampler(train_dts)

    # """""""""""""""""" Create Network """"""""""""""""""""""""""
    net = getter.get_model()
    scaler = torch.cuda.amp.GradScaler(enabled=config.model.net.with_autocast)

    if state is not None:
        scaler.load_state_dict(state['scaler_state'])
        net.load_state_dict(state['net_state'])
        net.cuda()

    # """""""""""""""""" Create Optimizer & Scheduler """"""""""""""""""""""""""
    optimizer, scheduler = getter.get_optimizer(net)
    if state is not None:
        for key, opt in optimizer.items():
            opt.load_state_dict(state['optimizer_state'][key])

        for key, sch in scheduler.items():
            sch.load_state_dict(state['scheduler_state'][key])

    # """""""""""""""""" Create Criterion """"""""""""""""""""""""""
    criterion = getter.get_loss()
    for crit, _ in criterion:
        if hasattr(crit, 'register_labels'):
            crit.register_labels(torch.from_numpy(train_dts.labels))

    memory = getter.get_memory()
    _ = None if memory is None else memory.cuda()

    if state is not None and "criterion_state" in state:
        for (crit, _), crit_state in zip(criterion, state["criterion_state"]):
            crit.cuda()
            crit.load_state_dict(crit_state)

    acc = getter.get_acc_calculator()

    assert not (config.model_parallel and config.distributed), "Only one must be choose between 'DataParallel' (model_parallel=True) and 'DistributedDataParallel' (distributed=True)"
    net.cuda()
    net = torch.nn.parallel.DataParallel(net) if config.model_parallel else net
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank) if config.distributed else net
    _ = [crit.cuda() for crit, _ in criterion]

    # """""""""""""""""" Handle RANDOM_STATE """"""""""""""""""""""""""
    if state is not None:
        # set random NumPy and Torch random states
        lib.set_random_state(state)

    return eng.train(
        config=config,
        log_dir=log_dir,
        is_master=is_master,
        net=net,
        criterion=criterion,
        memory=memory,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        acc=acc,
        train_dts=train_dts,
        test_dts=test_dts,
        sampler=sampler,
        writer=writer,
        restore_epoch=restore_epoch,
    )


if __name__ == '__main__':
    run()
