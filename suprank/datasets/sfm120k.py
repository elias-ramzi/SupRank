from typing import Optional, Callable, Mapping, Any, Type
from os.path import join

import numpy as np
from scipy.io import loadmat

from suprank.datasets.tools import filter_classes
from suprank.datasets.base_dataset import BaseDataset

import suprank.lib as lib

NoneType = Type[None]
KwargsType = Mapping[str, Any]


def cid2filename(cid: str, prefix: str) -> str:
    """
    https://github.com/filipradenovic/cnnimageretrieval-pytorch

    Creates a training image path out of its CID name

    Arguments
    ---------
    cid      : name of the image
    prefix   : root directory where images are saved

    Returns
    -------
    filename : full image filename
    """
    return join(prefix, cid[-2:], cid[-4:-2], cid[-6:-4], cid)


class SfM120kDataset(BaseDataset):

    HIERARCHY_LEVEL: int = 1

    def __init__(
        self,
        data_dir: str,
        mode: str = 'train',
        transform: Optional[Callable] = None,
        num_samples_th: Optional[int] = None,
        **kwargs: KwargsType,
    ) -> NoneType:
        self.data_dir = lib.expand_path(data_dir)
        self.mode = mode
        self.transform = transform
        self.num_samples_th = num_samples_th

        db = loadmat(join(self.data_dir, "retrieval-SfM-120k-imagenames-clusterids.mat"))

        cids = [x[0] for x in db['cids'][0]]
        self.paths = [cid2filename(x, join(self.data_dir, "ims")) for x in cids]
        self.labels = np.array([int(x) for x in db['cluster'][0]]).reshape(-1, 1)

        if self.num_samples_th is not None:
            accepted_index = filter_classes(self.labels, num_samples_th)
            self.labels = self.labels[accepted_index]
            self.paths = np.array(self.paths)[accepted_index].tolist()

        self.labels = lib.set_labels_to_range(self.labels)

        super().__init__(**kwargs)
