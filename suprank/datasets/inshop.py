from typing import Optional, Callable, Mapping, Any, Type
from os.path import join

import numpy as np

import suprank.lib as lib
from suprank.datasets.base_dataset import BaseDataset

NoneType = Type[None]
KwargsType = Mapping[str, Any]


class InShopDataset(BaseDataset):

    HIERARCHY_LEVEL: int = 3

    def __init__(
        self,
        data_dir: str,
        mode: str = 'train',
        transform: Optional[Callable] = None,
        hierarchy_mode: str = 'all',
        **kwargs: KwargsType,
    ) -> NoneType:
        assert mode in ["train", "query", "gallery"], f"Mode : {mode} unknown"
        assert hierarchy_mode in ['1', '2', 'all'], f"Hierarchy mode : {hierarchy_mode} unknown"
        self.data_dir = lib.expand_path(data_dir)
        self.mode = mode
        self.transform = transform

        with open(join(data_dir, "list_eval_partition.txt")) as f:
            db = f.read().split("\n")[2:-1]

        paths = []
        gender_name = []
        super_labels_name = []
        labels = []
        for line in db:
            line = line.split(" ")
            line = list(filter(lambda x: x, line))
            if line[2] == mode:
                paths.append(join(data_dir, line[0]))
                labels.append(int(line[1].split("_")[-1]))
                gender_name.append(line[0].split("/")[1])
                super_labels_name.append(line[0].split("/")[2])

        self.paths = paths

        slb_to_id = {slb: i for i, slb in enumerate(set(super_labels_name))}
        super_labels = [slb_to_id[slb] for slb in super_labels_name]

        gender_to_id = {gen: i for i, gen in enumerate(set(gender_name))}
        gender = [gender_to_id[gen] for gen in gender_name]

        if hierarchy_mode == 'all':
            self.labels = np.stack([labels, super_labels, gender], axis=1)

        elif hierarchy_mode == '1':
            self.labels = np.stack([labels, super_labels], axis=1)

        elif hierarchy_mode == '2':
            self.labels = np.stack([labels, gender], axis=1)

        else:
            raise ValueError

        self.labels = lib.set_labels_to_range(self.labels)
        super().__init__(**kwargs)

    @property
    def my_sub_repr(self,) -> str:
        return f"    hierarchy_mode={self.hierarchy_mode},\n"
