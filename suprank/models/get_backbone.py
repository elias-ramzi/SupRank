from typing import Tuple, Callable

import torch
import torch.nn as nn
import torchvision.models as models
import timm

import suprank.lib as lib


def get_backbone(name: str, pretrained: bool = True) -> Tuple[nn.Module, Callable, int]:
    if name == 'resnet18':
        lib.LOGGER.info("using ResNet-18")
        out_dim = 512
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    elif name == 'resnet34':
        lib.LOGGER.info("using ResNet-34")
        out_dim = 512
        backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    elif name == 'resnet50':
        lib.LOGGER.info("using ResNet-50")
        out_dim = 2048
        try:
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        except AttributeError:  # older versions of torchvision
            backbone = models.resnet50(pretrained=True if pretrained else None)
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    elif name == 'resnet101':
        lib.LOGGER.info("using ResNet-101")
        out_dim = 2048
        backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    elif name == 'vit_small':
        lib.LOGGER.info("using DeiT-S distilled")
        deit = timm.create_model('deit_small_patch16_224')
        deit_distilled = timm.create_model('deit_small_distilled_patch16_224', pretrained=pretrained)
        deit_distilled.pos_embed = nn.Parameter(torch.cat((deit_distilled.pos_embed[:, :1], deit_distilled.pos_embed[:, 2:]), dim=1))
        deit.load_state_dict(deit_distilled.state_dict(), strict=False)
        backbone = deit
        backbone.reset_classifier(-1)
        out_dim = 384
        pooling = nn.Identity()

    elif name == 'vit_base':
        lib.LOGGER.info("using DeiT-B distilled")
        deit = timm.create_model('deit_base_patch16_224')
        deit_distilled = timm.create_model('deit_base_distilled_patch16_224', pretrained=pretrained)
        deit_distilled.pos_embed = nn.Parameter(torch.cat((deit_distilled.pos_embed[:, :1], deit_distilled.pos_embed[:, 2:]), dim=1))
        deit.load_state_dict(deit_distilled.state_dict(), strict=False)
        backbone = deit
        backbone.reset_classifier(-1)
        out_dim = 768
        pooling = nn.Identity()

    elif name == 'vit_base_in21k':
        lib.LOGGER.info("using ViT-B with ImageNet21k pretraining")
        backbone = timm.create_model('vit_base_patch16_224_in21k', pretrained=pretrained)
        backbone.reset_classifier(-1)
        out_dim = 768
        pooling = nn.Identity()

    else:
        raise ValueError(f"{name} is not a valide backbone")

    return backbone, pooling, out_dim


if __name__ == '__main__':
    # This is usefull to preload backbones on a cluster (e.g. JZ)
    import logging

    DOWNLOAD_MODELS = [
        'resnet34',
        'resnet50',
        'resnet101',
        'vit_small',
        'vit_base',
    ]

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO,
    )

    for bck in DOWNLOAD_MODELS:
        _ = get_backbone(bck, True)
