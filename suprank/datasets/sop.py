from typing import Optional, Callable, Mapping, Any, Type
from os.path import join

import pandas as pd
import numpy as np

import suprank.lib as lib
from suprank.datasets.base_dataset import BaseDataset

NoneType = Type[None]
KwargsType = Mapping[str, Any]


class SOPDataset(BaseDataset):

    HIERARCHY_LEVEL: int = 2

    def __init__(
        self,
        data_dir: str,
        mode: str = 'train',
        transform: Optional[Callable] = None,
        **kwargs: KwargsType,
    ) -> NoneType:

        self.data_dir = lib.expand_path(data_dir)
        self.mode = mode
        self.transform = transform

        if mode == 'train':
            mode = ['train']
        elif mode == 'test':
            mode = ['test']
        elif mode == 'all':
            mode = ['train', 'test']
        else:
            raise ValueError(f"Mode unrecognized {mode}")

        self.paths = []
        labels = []
        super_labels = []
        for splt in mode:
            gt = pd.read_csv(join(self.data_dir, f'Ebay_{splt}.txt'), sep=' ')
            self.paths.extend(gt["path"].apply(lambda x: join(self.data_dir, x)).tolist())
            labels.extend((gt["class_id"] - 1).tolist())
            super_labels.extend((gt["super_class_id"] - 1).tolist())

        self.labels = np.stack([labels, super_labels], axis=1)

        self.super_label_names = [x.split('/')[-2][:-6] for x in self.paths]
        self.super_id_to_names = {}
        for id_, name in zip(self.labels[:, 1], self.super_label_names):
            self.super_id_to_names[id_] = name

        self.labels = lib.set_labels_to_range(self.labels)
        super().__init__(**kwargs)
