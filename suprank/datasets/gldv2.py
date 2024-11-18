from typing import Optional, Callable, Mapping, Any, Type
from os.path import join

import numpy as np
import pandas as pd

import suprank.lib as lib
from suprank.datasets.tools import filter_classes
from suprank.datasets.base_dataset import BaseDataset

NoneType = Type[None]
KwargsType = Mapping[str, Any]


def imgid2filename(imgid: str, prefix: str) -> str:
    return join(prefix, imgid[0], imgid[1], imgid[2], f"{imgid}.jpg")


class GLDV2Dataset(BaseDataset):

    HIERARCHY_LEVEL: int = 3

    def __init__(
        self,
        data_dir: str,
        mode: str = 'train',
        load_hierarchy: bool = True,
        remove_nan: bool = True,
        transform: Optional[Callable] = None,
        num_samples_th: Optional[int] = None,
        **kwargs: KwargsType,
    ) -> NoneType:
        assert mode in ['train', 'query', 'gallery']
        self.data_dir = lib.expand_path(data_dir)
        self.load_hierarchy = load_hierarchy
        self.remove_nan = remove_nan
        self.transform = transform
        self.mode = mode
        self.num_samples_th = num_samples_th
        if self.mode == 'train':
            self.folder = 'images/train'
            self.db_file = 'train_clean.csv'
            self.subcategory_file = 'train_label_to_subcategory_with_clusters.csv'
        elif self.mode == 'query':
            assert self.num_samples_th is None
            self.folder = 'test'
            self.db_file = 'retrieval_solution_v2.1.csv'
            self.subcategory_file = 'index_label_to_subcategory_with_clusters.csv'
        elif self.mode == 'gallery':
            assert self.num_samples_th is None
            self.folder = 'index'
            self.db_file = 'index_image_to_landmark.csv'
            self.subcategory_file = 'index_label_to_subcategory_with_clusters.csv'

        df = pd.read_csv(join(self.data_dir, self.db_file))
        if self.load_hierarchy:
            subcategory = pd.read_csv(join(self.data_dir, self.subcategory_file))[['landmark_id', 'cluster', 'overall_cluster']]
            self.cluster_to_id = {sc: scid for scid, sc in enumerate(sorted(subcategory['cluster'].dropna().unique()))}
            self.overall_cluster_to_id = {sc: scid for scid, sc in enumerate(sorted(subcategory['overall_cluster'].dropna().unique()))}
            subcategory['cluster_id'] = subcategory['cluster'].map(self.cluster_to_id)
            subcategory['overall_cluster_id'] = subcategory['overall_cluster'].map(self.overall_cluster_to_id)
        else:
            self.HIERARCHY_LEVEL = 1

        if self.mode == 'query':
            df = df[df['images'] != 'None'][['id', 'images']]
            landmark_id_df = pd.read_csv(join(self.data_dir, 'index_image_to_landmark.csv'))
            image_id_to_landmark_id = {iid: lid for iid, lid in landmark_id_df.values}
            df['landmark_id'] = df['images'].map(lambda x: image_id_to_landmark_id[x.split(" ")[0]])

        if self.load_hierarchy:
            df = df.merge(subcategory, on='landmark_id', how='left')
            if self.mode == 'train':
                df = df[['landmark_id', 'cluster_id', 'overall_cluster_id', 'images']]  # landmark_id in single row --> multiple images per row
            else:
                df = df[['landmark_id', 'cluster_id', 'overall_cluster_id', 'id']]  # one image per row
        else:
            if self.mode == 'train':
                df = df[['landmark_id', 'images']]  # landmark_id in single row --> multiple images per row
            else:
                df = df[['landmark_id', 'id']]  # one image per row

        self.paths = []
        self.labels = []
        for line in df.values:

            if self.load_hierarchy:
                landmark_id, cluster_id, overall_cluster_id, images = line[0], line[1], line[2], line[3].split(" ")
                for imgid in images:
                    self.labels.append([landmark_id, cluster_id, overall_cluster_id])
                    self.paths.append(imgid2filename(imgid, join(self.data_dir, self.folder)))
            else:
                landmark_id, images = line[0], line[1].split(" ")
                for imgid in images:
                    self.labels.append([landmark_id])
                    self.paths.append(imgid2filename(imgid, join(self.data_dir, self.folder)))

        self.labels = np.array(self.labels).reshape(len(self.paths), -1)

        if self.load_hierarchy and self.remove_nan:
            nan_cluster = np.isnan(self.labels[:, 1])
            self.labels = self.labels[~nan_cluster]
            self.paths = np.array(self.paths)[~nan_cluster].tolist()

        if self.mode == 'train':
            if self.num_samples_th is not None:
                accepted_index = filter_classes(self.labels, num_samples_th)
                self.labels = self.labels[accepted_index]
                self.paths = np.array(self.paths)[accepted_index].tolist()

            self.labels = lib.set_labels_to_range(self.labels.astype(np.int64))

        super().__init__(**kwargs)


if __name__ == '__main__':
    n_labels = 0

    dataset = GLDV2Dataset(
        data_dir='/local/SSD_DEEPLEARNING_2/image_retrieval/gldv2',
        mode='train',
        transform=None,
        num_samples_th=None,
    )
    print(dataset)
    print((~np.isnan(dataset.labels))[:, 1].sum())
    n_labels += len(np.unique(dataset.labels[~np.isnan(dataset.labels[:, 1]), 0]))
    print(n_labels)

    dataset = GLDV2Dataset(
        data_dir='/local/SSD_DEEPLEARNING_2/image_retrieval/gldv2',
        mode='gallery',
        transform=None,
        num_samples_th=None,
        compute_relevances=False,
    )
    print(dataset)
    print((~np.isnan(dataset.labels))[:, 1].sum())
    n_labels += len(np.unique(dataset.labels[~np.isnan(dataset.labels[:, 1]), 0]))
    print(n_labels)

    dataset = GLDV2Dataset(
        data_dir='/local/SSD_DEEPLEARNING_2/image_retrieval/gldv2',
        mode='query',
        transform=None,
        num_samples_th=None,
        compute_relevances=False,
    )
    print(dataset)
    print((~np.isnan(dataset.labels))[:, 1].sum())
