from typing import Mapping, Optional, Type
import os
import logging
import argparse

import torch
import numpy as np
from omegaconf import open_dict

import suprank.lib as lib
import suprank.engine as eng
from suprank.getter import Getter

NoneType = Type[None]


def print_metrics(metrics: Mapping[str, float]) -> NoneType:
    for split, mtrc in metrics.items():
        for k, v in mtrc.items():
            if k == 'epoch':
                continue
            lib.LOGGER.info(f"{split} --> {k} : {np.around(v*100, decimals=2)}")
        if split in ['test']:
            lib.LOGGER.info(f"This is for latex--> {lib.format_for_latex(mtrc)}")
        if split in ['test_overall']:
            lib.LOGGER.info(f"This is for DyLM latex--> {lib.format_for_latex_dyml(mtrc)}")
        print()
        print()


def load_and_evaluate(
    path: str,
    hierarchy_level: str,
    set: str,
    relevance_type: str,
    factor: float,
    ibs: int,
    mbs: int,
    nw: int,
    data_dir: Optional[str] = None,
    no_amp: bool = False,
    inat_base: bool = False,
    inat_full: bool = False,
    pretrained: bool = False,
) -> Mapping[str, float]:
    os.environ['USE_CUDA_FOR_RELEVANCE'] = 'yes'
    lib.LOGGER.info(f"Evaluating : \033[92m{path}\033[0m")
    state = torch.load(lib.expand_path(path), map_location='cpu')
    cfg = state["config"]

    if pretrained:
        cfg.model.net.norm_features = False
        cfg.model.net.without_fc = True
        cfg.model.net.pretrained = True
        cfg.model.net.pooling = 'default'

    # TEMP:
    cfg.evaluation.exclude = []
    with open_dict(cfg.evaluation):
        cfg.evaluation.with_binary_asi = True

    if factor:
        cfg.dataset.dts.train.factor = factor[0]
    if relevance_type:
        cfg.dataset.dts.train.relevance_type = relevance_type

    lib.LOGGER.info("Loading model...")
    cfg.model.net.with_autocast = not no_amp
    net = Getter(cfg).get_model()
    if not pretrained:
        net.load_state_dict(lib.adapt_checkpoint(state["net_state"]))
    net.cuda()
    net.eval()

    if data_dir is not None:
        cfg.dataset.dts.train.data_dir = lib.expand_path(data_dir)

    if inat_base:
        assert not inat_full
        if hasattr(cfg.dataset.dts.train, 'hierarchy_mode'):
            cfg.dataset.dts.train.hierarchy_mode = 'base'
        else:
            with open_dict(cfg.dataset.dts.train):
                cfg.dataset.dts.train.hierarchy_mode = 'base'
        hierarchy_level = [0, 1]

    if inat_full:
        assert not inat_base
        if hasattr(cfg.dataset.dts.train, 'hierarchy_mode'):
            cfg.dataset.dts.train.hierarchy_mode = 'full'
        else:
            with open_dict(cfg.dataset.dts.train):
                cfg.dataset.dts.train.hierarchy_mode = 'full'
        hierarchy_level = [0, 1, 2, 3, 4, 5, 6]

    getter = Getter(cfg)
    datasets = getter.get_dataset()
    _, dts = datasets['train'], datasets['test']

    if set == 'test':
        dataset_dict = {}
        if isinstance(dts, list):
            for i, _dts in enumerate(dts):
                dataset_dict[f"test_level{i}"] = _dts
        else:
            dataset_dict["test"] = dts
    else:
        dataset_dict = {set: dts}

    lib.LOGGER.info("Dataset created...")

    cfg.evaluation.num_workers = nw
    cfg.evaluation.inference_batch_size = ibs
    cfg.evaluation.metric_batch_size = mbs
    cfg.evaluation.recall_rate = [1, 2, 4, 8, 10, 16, 32, 100]
    if hierarchy_level is not None:
        cfg.evaluation.compute_for_hierarchy_levels = hierarchy_level
    getter = Getter(cfg)
    acc = getter.get_acc_calculator()

    metrics = eng.evaluate(
        net=net,
        dataset_dict=dataset_dict,
        acc=acc,
        epoch=state["epoch"],
    )

    lib.LOGGER.info("Evaluation completed...")
    print_metrics(metrics)

    return metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help='Path to checkpoint')
    parser.add_argument("--hierarchy-level", type=int, default=None, nargs='+', help='Hierarchy level for Acc')
    parser.add_argument("--set", type=str, default='test', help='Set on which to evaluate')
    parser.add_argument("--relevance-type", type=str, default=None, help='Relevance type')
    parser.add_argument("--factor", type=float, nargs='+', default=None, help='Factor to compute the H-AP and NDCG')
    parser.add_argument("--iBS", type=int, default=256, help='Batch size for DataLoader')
    parser.add_argument("--mBS", type=int, default=256, help='Batch size for metric calculation')
    parser.add_argument("--nw", type=int, default=10, help='Num workers for DataLoader')
    parser.add_argument("--data-dir", type=str, default=None, help='Possible override of the datadir in the dataset config')
    parser.add_argument("--no-amp", default=True, action='store_false', help='Deactivates mix precision')
    parser.add_argument("--metrics-from-checkpoint", default=False, action='store_true', help='Only reads the metrics in the checkpoint')
    parser.add_argument("--inat-base", default=False, action='store_true', help='Allow the use of a model trained on iNat-Full to be evaluated on iNat-Base')
    parser.add_argument("--inat-full", default=False, action='store_true', help='Allow the use of a model trained on iNat-Full to be evaluated on iNat-Base')
    parser.add_argument("--pretrained", default=False, action='store_true', help='Use a config to evalute the pretrained model')
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO,
    )

    if args.metrics_from_checkpoint:
        metrics = torch.load(args.config, map_location='cpu')['metrics']
        lib.LOGGER.info(f"READING : \033[92m{args.config}\033[0m")
        print_metrics(metrics)
    else:
        metrics = load_and_evaluate(
            path=args.config,
            hierarchy_level=args.hierarchy_level,
            set=args.set,
            relevance_type=args.relevance_type,
            factor=args.factor,
            ibs=args.iBS,
            mbs=args.mBS,
            nw=args.nw,
            data_dir=args.data_dir,
            no_amp=args.no_amp,
            inat_base=args.inat_base,
            inat_full=args.inat_full,
            pretrained=args.pretrained,
        )
        print()
        print()
