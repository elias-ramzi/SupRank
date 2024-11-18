"""
Adapted from:
https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/utils/evaluate.py
"""
from typing import List, Mapping, Optional, Type

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import suprank.lib as lib

NoneType = Type[None]


@torch.no_grad()
def compute_ap(ranks: np.ndarray, nres: int) -> float:
    """
    Computes average precision for given ranked indexes.

    Arguments
    ---------
    ranks : zerro-based ranks of positive images
    nres  : number of positive images

    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap


@torch.no_grad()
def compute_map(ranks: np.ndarray, gnd: np.ndarray) -> float:
    """
    Computes the mAP for a given set of returned results.
         Usage:
           map = compute_map (ranks, gnd)
                 computes mean average precsion (map) only

           map, aps, pr, prs = compute_map (ranks, gnd, kappas)
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query

         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    map = 0.
    nq = len(gnd)  # number of queries
    aps = np.zeros(nq)
    # pr = np.zeros(len(kappas))
    # prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            # prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except Exception:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgndj)]

        k = 0
        ij = 0
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # # compute precision @ k
        # pos += 1  # get it to 1-based
        # for j in np.arange(len(kappas)):
        #     kq = min(max(pos), kappas[j])
        #     prs[i, j] = (pos <= kq).sum() / kq
        # pr = pr + prs[i, :]

    map = map / (nq - nempty)
    # pr = pr / (nq - nempty)

    return map  # , aps, pr, prs


@torch.no_grad()
def compute_map_M_and_H(ranks: np.ndarray, gnd: List[Mapping[str, List[int]]]) -> Mapping[str, float]:

    # gnd_t = []
    # for i in range(len(gnd)):
    #     g = {}
    #     g['ok'] = np.concatenate([gnd[i]['easy']])
    #     g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
    #     gnd_t.append(g)
    # mapE, *_ = compute_map(ranks, gnd_t, kappas)

    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
        g['junk'] = np.concatenate([gnd[i]['junk']])
        gnd_t.append(g)
    mapM = compute_map(ranks, gnd_t)

    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['hard']])
        g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
        gnd_t.append(g)
    mapH = compute_map(ranks, gnd_t)

    return {"mapM": mapM, "mapH": mapH}


@torch.no_grad()
def evaluate_a_city(
    net: nn.Module,
    query: Dataset,
    gallery: Dataset,
    batch_size: int = 256,
    num_workers: int = 10,
    amp: bool = False,
    multi_scale: bool = False,
) -> Mapping[str, float]:

    def collate_fn(batch: List[Mapping[str, Tensor]]) -> Mapping[str, Tensor]:
        out = {}
        out["image"] = torch.stack([b["image"] for b in batch], dim=0)
        out["label"] = torch.cat([b["label"] for b in batch])
        return out

    scales = [1, 1 / np.sqrt(2), 1 / 2] if multi_scale else None

    features_query = []
    features_gallery = []
    loader_query = DataLoader(query, batch_size=batch_size, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    loader_gallery = DataLoader(gallery, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    lib.LOGGER.info("Computing embeddings")
    for batch in lib.track(loader_gallery, "Gallery embedddings"):
        with torch.cuda.amp.autocast(enabled=amp):
            X = net(batch["image"].to('cuda', non_blocking=True), scales=scales)
        features_gallery.append(X.float())

    features_gallery = torch.cat(features_gallery)

    for batch in lib.track(loader_query, "Query embeddings"):
        with torch.cuda.amp.autocast(enabled=amp):
            X = net(batch["image"].to('cuda', non_blocking=True), scales=scales)
        features_query.append(X.float())

    features_query = torch.cat(features_query)

    features_gallery = features_gallery.cpu().numpy()
    features_query = features_query.cpu().numpy()

    # search, rank, and print
    scores = np.dot(features_gallery, features_query.T)
    ranks = np.argsort(-scores, axis=0)

    return compute_map_M_and_H(ranks, query)


class LandmarkEvaluation:

    def __init__(
        self,
        batch_size: int = 256,
        num_workers: int = 10,
        amp: bool = False,
        multi_scale: bool = False,
    ) -> NoneType:
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.amp = amp
        self.multi_scale = multi_scale

    @torch.no_grad()
    def evaluate(
        self,
        net: nn.Module,
        datasets: List[Mapping[str, Dataset]],
        epoch: Optional[int] = None,
    ) -> Mapping[str, Mapping[str, float]]:
        metrics = {}
        for city_name, dts in datasets["test"].items():
            metrics[city_name] = evaluate_a_city(
                net=net,
                query=dts['query'],
                gallery=dts['gallery'],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                amp=self.amp,
                multi_scale=self.multi_scale,
            )

        return metrics

    def __repr__(self) -> str:
        repr = f"{self.__class__.__name__}(\n"
        repr += f"    batch_size={self.batch_size},\n"
        repr += f"    num_workers={self.num_workers},\n"
        repr += f"    amp={self.amp},\n"
        repr += f"    multi_scale={self.multi_scale},\n"
        repr += ")"
        return repr
