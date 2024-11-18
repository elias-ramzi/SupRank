from typing import Optional, Any

import torch
from omegaconf.dictconfig import DictConfig

from suprank.lib.expand_path import expand_path


def load_state(path: str, key: Optional[str] = None) -> Any:  # noqa ANN401
    state = torch.load(expand_path(path), map_location='cpu')

    if key is not None:
        return state[key]

    return state


def load_config(path: str) -> DictConfig:
    return load_state(path, 'config')
