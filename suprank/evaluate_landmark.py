from typing import Mapping, Optional, Type
import os
import logging
import argparse
from functools import partial

import torch
import torchvision.transforms as T
import numpy as np

import suprank.lib as lib
import suprank.engine as eng
from suprank.getter import Getter
from suprank.datasets import RevisitedDataset, GLDV2Dataset
from suprank.engine import AccuracyCalculator, LandmarkEvaluation, hap_LM

NoneType = Type[None]


def print_metrics(metrics_benchmark: Optional[Mapping[str, float]] = None, metrics_gldv2: Optional[Mapping[str, float]] = None) -> NoneType:
    for_latex = ""

    if metrics_benchmark is not None:
        for city, mtrc in metrics_benchmark.items():
            for k, v in mtrc.items():
                lib.LOGGER.info(f"{city} --> {k} : {np.around(v*100, decimals=2)}")

        for_latex += f"& {np.around(100 * metrics_benchmark['ROxford5k']['mapM'], decimals=2)} "
        for_latex += f"& {np.around(100 * metrics_benchmark['ROxford5k']['mapH'], decimals=2)} "
        for_latex += f"& {np.around(100 * metrics_benchmark['RParis6k']['mapM'], decimals=2)} "
        for_latex += f"& {np.around(100 * metrics_benchmark['RParis6k']['mapH'], decimals=2)} "

    if metrics_gldv2 is not None:
        for k, v in metrics_gldv2['test'].items():
            lib.LOGGER.info(f"GLDv2 --> {k} : {np.around(v*100, decimals=2)}")

        for_latex += f"& {np.around(100 * metrics_gldv2['test']['mAP@100_level0'], decimals=2)} "
        for_latex += f"& {np.around(100 * metrics_gldv2['test']['H-AP_multi'], decimals=2)} "
        for_latex += f"& {np.around(100 * metrics_gldv2['test']['ASI_multi'], decimals=2)} "
        for_latex += f"& {np.around(100 * metrics_gldv2['test']['NDCG_multi'], decimals=2)} "

    for_latex += "\\\\"  # end of line
    lib.LOGGER.info(f"This is for latex --> {for_latex}")


def load_and_evaluate(
    path: str,
    relevance_type: str,
    factor: float,
    imsize: int,
    multi_scale: bool,
    ibs: int,
    mbs: int,
    nw: int,
    data_dir: Optional[str] = None,
    model_parallel: bool = False,
    no_amp: bool = False,
) -> Mapping[str, float]:
    os.environ['USE_CUDA_FOR_RELEVANCE'] = 'yes'
    lib.LOGGER.info(f"Evaluating : \033[92m{path}\033[0m")
    state = torch.load(lib.expand_path(path), map_location='cpu')
    cfg = state["config"]

    if factor:
        cfg.dataset.dts.train.factor = factor[0]
    if relevance_type:
        cfg.dataset.dts.train.relevance_type = relevance_type

    lib.LOGGER.info("Loading model...")
    cfg.model.net.with_autocast = not no_amp
    net = Getter(cfg).get_model()
    net.load_state_dict(lib.adapt_checkpoint(state["net_state"]))
    net.cuda()
    net.eval()
    net.requires_grad_(False)
    if model_parallel:
        lib.LOGGER.info("Model parallelism enabled")
        net = torch.nn.DataParallel(net)

    if data_dir is not None:
        cfg.dataset.dts.train.data_dir = lib.expand_path(data_dir)

    metrics_benchmark, metrics_gldv2 = None, None

    if not args.no_lm:
        dts_benchmark = {
            'RParis6k': {
                'query':
                    RevisitedDataset(
                        data_dir='/local/SSD_DEEPLEARNING_1/image_retrieval/rparis6k/',
                        mode='query',
                        compute_relevances=False,
                        transform=T.Compose([
                            T.Resize((imsize, imsize)) if args.imsize > 0 else T.Lambda(lambda x: x),
                            T.ToTensor(),
                            T.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225],
                            ),
                        ]),
                    ),
                'gallery':
                    RevisitedDataset(
                        data_dir='/local/SSD_DEEPLEARNING_1/image_retrieval/rparis6k/',
                        mode='gallery',
                        compute_relevances=False,
                        transform=T.Compose([
                            T.Resize((imsize, imsize)) if args.imsize > 0 else T.Lambda(lambda x: x),
                            T.ToTensor(),
                            T.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225],
                            ),
                        ]),
                    ),
            },
            'ROxford5k': {
                'query':
                RevisitedDataset(
                    data_dir='/local/SSD_DEEPLEARNING_1/image_retrieval/roxford5k/',
                    mode='query',
                    compute_relevances=False,
                    transform=T.Compose([
                        T.Resize((imsize, imsize)) if args.imsize > 0 else T.Lambda(lambda x: x),
                        T.ToTensor(),
                        T.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        ),
                    ]),
                ),
                'gallery':
                    RevisitedDataset(
                        data_dir='/local/SSD_DEEPLEARNING_1/image_retrieval/roxford5k/',
                        mode='gallery',
                        compute_relevances=False,
                        transform=T.Compose([
                            T.Resize((imsize, imsize)) if args.imsize > 0 else T.Lambda(lambda x: x),
                            T.ToTensor(),
                            T.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225],
                            ),
                        ]),
                ),
            },
        }

        acc_benchmark = LandmarkEvaluation(
            batch_size=1,
            num_workers=nw,
            amp=not no_amp,
            multi_scale=multi_scale,
        )
        metrics_benchmark = eng.evaluate(
            net=net,
            dataset_dict={'test': dts_benchmark},
            acc=acc_benchmark,
            epoch=state["epoch"],
        )
        print(metrics_benchmark)

    if not args.no_gldv2:
        dts_gldv2 = {
            'query':
                GLDV2Dataset(
                    data_dir="/local/DEEPLEARNING/image_retrieval/gldv2/",
                    mode='query',
                    compute_relevances=False,
                    transform=T.Compose([
                        T.Resize((imsize, imsize)) if args.imsize > 0 else T.Lambda(lambda x: x),
                        T.ToTensor(),
                        T.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        ),
                    ]),
                ),
            'gallery':
                GLDV2Dataset(
                    data_dir="/local/DEEPLEARNING/image_retrieval/gldv2/",
                    mode='gallery',
                    compute_relevances=False,
                    transform=T.Compose([
                        T.Resize((imsize, imsize)) if args.imsize > 0 else T.Lambda(lambda x: x),
                        T.ToTensor(),
                        T.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        ),
                    ]),
                ),
        }

        acc_gldv2 = AccuracyCalculator(
            compute_for_hierarchy_levels=[0],
            precision_rate=[100],
            map_at_k=[100],
            inference_batch_size=ibs,
            metric_batch_size=mbs,
            num_workers=nw,
        )
        acc_gldv2.METRICS_DICT['multi_level']['H-AP'] = partial(hap_LM, hierarchy_levels=dts_gldv2['query'].HIERARCHY_LEVEL)

        metrics_gldv2 = eng.evaluate(
            net=net,
            dataset_dict={'test': dts_gldv2},
            acc=acc_gldv2,
            epoch=state["epoch"],
        )
        print(metrics_gldv2)

    lib.LOGGER.info("Evaluation completed...")
    print_metrics(metrics_benchmark, metrics_gldv2)

    return metrics_benchmark, metrics_gldv2


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help='Path to checkpoint')
    parser.add_argument("--parse-file", default=False, action='store_true', help='If passed the config file will be considered as a .txt file and parsed')
    parser.add_argument("--relevance-type", type=str, default=None, help='Relevance type')
    parser.add_argument("--factor", type=float, nargs='+', default=None, help='Factor to compute the H-AP and NDCG')
    parser.add_argument("--imsize", type=int, default=224, help='Batch size for DataLoader')
    parser.add_argument("--iBS", type=int, default=256, help='Batch size for DataLoader')
    parser.add_argument("--mBS", type=int, default=64, help='Batch size for metric calculation')
    parser.add_argument("--nw", type=int, default=10, help='Num workers for DataLoader')
    parser.add_argument("--data-dir", type=str, default=None, help='Possible override of the datadir in the dataset config')
    parser.add_argument("--multi-scale", default=False, action='store_true', help='Run multi scale evaluation for R-Paris6k and R-Oxford5k')
    parser.add_argument("--no-lm", default=False, action='store_true', help='Deactivate R-Paris6k and R-Oxford5k evaluation')
    parser.add_argument("--no-gldv2", default=False, action='store_true', help='Deactivate GLDv2 evaluation')
    parser.add_argument("--no-amp", default=True, action='store_false', help='Deactivates mix precision')
    parser.add_argument("--model-parallel", default=False, action='store_true', help='Parallelize the model across GPUs')
    parser.add_argument("--metrics-from-checkpoint", default=False, action='store_true', help='Only reads the metrics in the checkpoint')
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO,
    )

    if args.parse_file:
        with open(args.config, 'r') as f:
            config = f.readlines()
        config = [cfg.replace('\n', '') for cfg in config if cfg]
    else:
        config = [args.config]

    for cfg in config:
        args.config = cfg

        if args.metrics_from_checkpoint:
            metrics = torch.load(args.config, map_location='cpu')['metrics']
            lib.LOGGER.info(f"READING : \033[92m{args.config}\033[0m")
            print_metrics(metrics)
        else:
            metrics_benchmark, metrics_gldv2 = load_and_evaluate(
                path=args.config,
                relevance_type=args.relevance_type,
                factor=args.factor,
                imsize=args.imsize,
                multi_scale=args.multi_scale,
                ibs=args.iBS,
                mbs=args.mBS,
                nw=args.nw,
                data_dir=args.data_dir,
                model_parallel=args.model_parallel,
                no_amp=args.no_amp,
            )
            print()
            print()
