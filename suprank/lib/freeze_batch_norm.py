import torch.nn as nn


def freeze_batch_norm(model: nn.Module) -> nn.Module:
    for module in filter(lambda m: isinstance(m, nn.BatchNorm2d), model.modules()):
        module.eval()
        module.train = lambda _: None
    return model
