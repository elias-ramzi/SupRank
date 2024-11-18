from typing import Optional, Callable, Mapping, Any, Type
from os.path import join

import pandas as pd

import suprank.lib as lib
from suprank.datasets.base_dataset import BaseDataset

NoneType = Type[None]
KwargsType = Mapping[str, Any]


class DyMLDataset(BaseDataset):
    HIERARCHY_LEVEL: int = 3

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

        name = self.data_dir.split('/')[1:]
        if not name[-1]:
            name.remove("")
        name = name[-1].split('_')[-1]
        self.__class__.__name__ = f"DyML{name.title()}"

        if mode == 'train':
            table = pd.read_csv(join(self.data_dir, 'train', 'label.csv'))
            paths = table["fname"].tolist()
            self.paths = [join(self.data_dir, 'train', 'imgs', x) for x in paths]

            labels = table[[" fine_id0.jpg", " middle_id", " coarse_id"]].to_numpy()

            self.labels = lib.set_labels_to_range(labels)

        elif mode.startswith("test"):
            # mode is for example 'test_query_fine'
            _, type, granularity = mode.split("_")
            assert type in ['query', 'gallery']
            assert granularity in ['fine', 'middle', 'coarse']
            table = pd.read_csv(join(self.data_dir, f'bmk_{granularity}', f'{type}.csv'))
            paths = table["fname"].tolist()
            labels = table[" labels0.jpg"].to_numpy().reshape(-1, 1)

            self.paths = [join(self.data_dir, f"bmk_{granularity}", f"{type}", x) for x in paths]
            self.labels = labels
            self.labels = lib.set_labels_to_range(self.labels)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        super().__init__(**kwargs)


class DyMLProduct(BaseDataset):
    HIERARCHY_LEVEL = 3

    def __init__(
        self,
        data_dir: str,
        mode: str,
        transform: Optional[Callable] = None,
        **kwargs: KwargsType,
    ) -> NoneType:

        self.data_dir = lib.expand_path(data_dir)
        self.mode = mode
        self.transform = transform

        if mode == 'train':
            table = pd.read_csv(join(self.data_dir, 'train', 'label.csv'))
            paths = table["fname"].tolist()
            self.paths = [join(self.data_dir, 'train', 'imgs', x) for x in paths]

            labels = table[[" fine_id0.jpg", " middle_id", " coarse_id"]].to_numpy()

            self.labels = lib.set_labels_to_range(labels)

        elif mode == 'test':
            table = pd.read_csv(join(self.data_dir, "mini-bmk_all_in_one", 'label.csv'))
            paths = table["fname"].tolist()
            self.paths = [join(self.data_dir, "mini-bmk_all_in_one", "imgs", x) for x in paths]

            self.labels = table[[" fine_id0.jpg", " middle_id", " coarse_id"]].to_numpy()

        else:
            raise ValueError(f"Unknown mode: {mode}")

        super().__init__(**kwargs)
