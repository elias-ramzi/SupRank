from typing import List, Any, Mapping, Tuple, Type
import os
import builtins
import subprocess

import torch

from suprank.lib.logger import LOGGER

NoneType = Type[None]


def set_distributed(distributed: bool, is_cluster: bool) -> Tuple[int]:
    if is_cluster and distributed:
        # local rank on the current node / global rank
        local_rank = int(os.environ['SLURM_LOCALID'])
        global_rank = int(os.environ['SLURM_PROCID'])
        # number of processes / GPUs per node
        world_size = int(os.environ['SLURM_NTASKS'])
        # define master address and master port
        hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
        master_addr = hostnames.split()[0].decode('utf-8')
        # set environment variables for 'env://'
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(29500)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(global_rank)
        os.environ['LOCAL_RANK'] = str(local_rank)

    if distributed:
        torch.distributed.init_process_group(backend='nccl')
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        world_size = 1
        rank = 0
        local_rank = 0
    is_master = rank == 0
    torch.cuda.set_device(local_rank)

    if distributed and (not is_master):
        def print_pass(*args: List[Any], force: bool = False, **kwargs: Mapping[str, Any]) -> NoneType:
            if force:
                print(*args, **kwargs)
        builtins.print = print_pass
        LOGGER.propagate = False
        os.environ['TQDM_DISABLE'] = 'true'

    return is_master, local_rank
