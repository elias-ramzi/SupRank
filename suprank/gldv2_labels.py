from typing import Tuple, Callable, List, Mapping, Type
import os
import argparse
import logging
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
import open_clip

import suprank.lib as lib
from suprank.datasets import GLDV2Dataset

NoneType = Type[None]


QUERIES = [
    "This is a picture of a {}",
]


def get_model(model_name: str = 'ViT-L-14', pretrained: str = 'laion2b_s32b_b82k') -> Tuple[nn.Module, transforms.Compose, Callable]:
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    _ = model.requires_grad_(False)
    _ = model.eval()
    _ = model.cuda()
    tokenizer = open_clip.get_tokenizer(model_name)

    return model, preprocess, tokenizer


def get_dataset(
        data_dir: str,
        mode: str = 'train',
        transform: transforms.Compose = None,
) -> Dataset:

    return GLDV2Dataset(
        data_dir,
        mode=mode,
        transform=transform,
        compute_relevances=False,
    )


def get_loader(dataset: Dataset) -> DataLoader:
    return DataLoader(dataset, batch_size=128, num_workers=5, pin_memory=True, shuffle=False, drop_last=False)


def get_clusters(csv_dir: str) -> Tuple[List[str], Tensor, Mapping[str, str], Mapping[str, int], Mapping[int, str]]:

    cluster_clean = pd.read_csv(csv_dir)

    def is_not_int(x: str) -> bool:
        x = x.replace('1\u202f222,00', "1222")
        x = x.replace(',', '.')

        try:
            float(x)
            return False
        except ValueError:
            return True

    keep_col = list(filter(is_not_int, cluster_clean.columns.tolist()))
    assert len(keep_col) == len(cluster_clean.columns) // 2

    cluster_clean = cluster_clean[keep_col]
    cluster_clean = cluster_clean.to_dict(orient='list')

    cluster_mapping = {}
    for clu, sub_categories in cluster_clean.items():
        for sc in sub_categories:
            if sc == 'neighborhood park':
                sc = 'neighborhood  park'
            elif sc == 'quarter of hamburg':
                sc = 'quarter of  hamburg'

            cluster_mapping[sc] = clu

    del cluster_mapping[np.nan]

    names = sorted(list(cluster_mapping.keys()))

    clusters_to_id = {sc: scid for scid, sc in enumerate(sorted(set(cluster_mapping.values())))}
    id_to_clusters = {iclu: clu for clu, iclu in clusters_to_id.items()}

    text_labels = torch.tensor([
        clusters_to_id[cluster_mapping[x]] for x in names
    ])

    return names, text_labels, clusters_to_id, cluster_mapping, id_to_clusters


def get_visual_features(
        model: nn.Module,
        loader: DataLoader,
) -> Tensor:
    visual_features = torch.empty(len(loader.dataset), 768, dtype=torch.float32, device='cuda')

    for i, batch in enumerate(lib.track(loader, "Computing visual features")):
        with torch.no_grad(), torch.cuda.amp.autocast():
            X = model.encode_image(batch['image'].to('cuda', non_blocking=True))
            X /= X.norm(dim=-1, keepdim=True)

        visual_features[i * loader.batch_size: (i + 1) * loader.batch_size] = X.float()

    return visual_features


def get_text_features(
        model: nn.Module,
        names: DataLoader,
        tokenizer: Callable,
) -> Tensor:
    text_features = torch.empty(len(names), 768, dtype=torch.float32, device='cuda')

    BS = 256
    iterator = range(len(names) // BS + 1)
    for i in lib.track(iterator, "Computing text features"):
        text = [f"A picture of a {x}" for x in names[i * BS: (i + 1) * BS]]
        text = tokenizer(text).to('cuda', non_blocking=True)

        with torch.no_grad(), torch.cuda.amp.autocast():
            text = model.encode_text(text)
            text /= text.norm(dim=-1, keepdim=True)

        text_features[i * BS: (i + 1) * BS] = text.float()

    return text_features


def clip_style_classification(
        visual_features: Tensor,
        text_features: Tensor,
        text_labels: Tensor,
        temperature: float = 100,
) -> Tensor:
    text_probs = (temperature * visual_features @ text_features.T).softmax(dim=-1)
    preds = text_probs.argmax(dim=-1)
    return text_labels.gather(0, preds.cpu()).view(-1).cpu()


def knn_style_classification(
        visual_features: Tensor,
        text_features: Tensor,
        text_labels: Tensor,
        topk: int = 5,
        temperature: float = 100,
) -> Tensor:
    topk = 5
    num_classes = len(text_labels.unique())
    retrieval_one_hot = torch.zeros(topk, num_classes).to('cuda', non_blocking=True)

    scores = visual_features @ text_features.T
    yd, yi = scores.topk(topk, -1)

    batch_size = len(scores)

    candidates = text_labels.view(1, -1).expand(batch_size, -1)
    retrieval = torch.gather(candidates, 1, yi.cpu()).long()
    retrieval = retrieval.to('cuda', non_blocking=True)
    retrieval_one_hot.resize_(batch_size * topk, num_classes).zero_()
    retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
    yd_transform = yd.clone().div_(1 / temperature).exp_()
    probs = torch.sum(torch.mul(retrieval_one_hot.view(batch_size, -1, num_classes), yd_transform.view(batch_size, -1, 1)), 1)

    return probs.argmax(dim=-1).view(-1).tolist()


def create_final_labels(
        preds: Tensor,
        labels: np.ndarray,
        unlabled_idx: np.ndarray,
) -> np.ndarray:
    flagged_cluster_per_labels = defaultdict(list)
    for i in range(len(unlabled_idx)):
        idx = unlabled_idx[i]
        lb = labels[idx, 0]
        assert labels[idx, 1] != labels[idx, 1]
        flagged_cluster_per_labels[lb].append(preds[i])

    final_cluster_per_labels = {}
    for lb, cls in flagged_cluster_per_labels.items():
        final_cluster_per_labels[lb] = Counter(cls).most_common(1)[0][0]

    final_labels = labels.copy()
    for i in range(len(unlabled_idx)):
        idx = unlabled_idx[i]
        lb = labels[idx, 0]
        clu = final_cluster_per_labels[lb]
        final_labels[idx, 1] = clu

    return final_labels


def main(args: argparse.Namespace) -> NoneType:
    model, preprocess, tokenizer = get_model()

    names, text_labels, clusters_to_id, cluster_mapping, id_to_clusters = get_clusters(args.cluster_dir)

    dataset = get_dataset(args.data_dir, mode=args.set, transform=preprocess)

    super_labels = np.array([clusters_to_id.get(cluster_mapping.get(x, np.nan), np.nan) for x in dataset.super_label_names])
    labels = np.stack((dataset.labels[:, 0:1].flatten(), super_labels.flatten()), axis=1)

    unlabeled_indices = np.where(np.isnan(labels[:, 1]))[0]
    sub_dataset = Subset(dataset, unlabeled_indices)

    loader = get_loader(sub_dataset)

    visual_features = get_visual_features(model, loader)
    text_features = get_text_features(model, names, tokenizer)

    if args.classification == 'clip':
        preds = clip_style_classification(visual_features, text_features, text_labels)
    elif args.classification == 'knn':
        preds = knn_style_classification(visual_features, text_features, text_labels)
    else:
        raise ValueError(f"Unknown classification method '{args.classification}'")

    final_labels = create_final_labels(preds, labels, unlabeled_indices)

    np.save(
        os.path.join(args.data_dir, f"{args.set}_labels.npy"),
        final_labels,
    )

    lib.save_json(id_to_clusters, os.path.join(args.data_dir, f"{args.set}_id_to_clusters.json"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", type=str, required=True, help='Set on which to evaluate')
    parser.add_argument("--data_dir", type=str, required=True, help='Directory containing the data')
    parser.add_argument("--cluster_dir", type=str, required=True, help='Directory containing the clusters')
    parser.add_argument("--gpu", type=int, default='all', help='Set the GPU to use')
    parser.add_argument("--classification", type=str, default='clip', help="Classification method to use ('knn' or 'clip')")
    args = parser.parse_args()

    if args.gpu != 'all':
        torch.cuda.set_device(args.gpu)

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO,
    )

    assert args.set in ['train', 'query', 'gallery', 'all', 'none']
    assert args.classification in ['clip', 'knn']

    if args.set == 'all':
        modes = ['train', 'query', 'gallery']
    elif args.set == 'none':
        modes = []
    else:
        modes = [args.set]

    for md in modes:
        args.set = md
        main(args)

    if args.set == 'none':
        query_labels = np.load(os.path.join(args.data_dir, 'query_labels.npy')).astype(np.int64)
        gallery_labels = np.load(os.path.join(args.data_dir, 'gallery_labels.npy')).astype(np.int64)
        query_labels = np.unique(query_labels, axis=0)
        tmp_gallery_labels = np.unique(gallery_labels, axis=0)

        diff_slb = []
        for i, lb in enumerate(query_labels[:, 0]):
            q_slb = query_labels[i, 1]
            g_slb = tmp_gallery_labels[tmp_gallery_labels[:, 0] == lb, 1]
            if q_slb != g_slb:
                diff_slb.append([lb, q_slb])

        print(len(diff_slb))

        if len(diff_slb) > 0:
            final_gallery_labels = gallery_labels.copy()
            for i in range(len(diff_slb)):
                lb = diff_slb[i][0]
                slb = diff_slb[i][1]
                final_gallery_labels[gallery_labels[:, 0] == lb, 1] = slb

            os.rename(os.path.join(args.data_dir, 'gallery_labels.npy'), os.path.join(args.data_dir, 'gallery_labels_old.npy'))
            np.save(os.path.join(args.data_dir, 'gallery_labels.npy'), final_gallery_labels)
