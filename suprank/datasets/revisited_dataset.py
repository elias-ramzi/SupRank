from typing import Optional, Callable, Mapping, Any, List, Type
from os.path import join

import torch
import pickle
from PIL import Image

import suprank.lib as lib
from suprank.datasets.base_dataset import BaseDataset

NoneType = Type[None]
KwargsType = Mapping[str, Any]


def imresize(img: Image.Image, imsize: int) -> Image.Image:
    img.thumbnail((imsize, imsize), Image.ANTIALIAS)
    return img


def path_to_label(pth: str) -> str:
    return "_".join(pth.split("/")[-1].split(".")[0].split("_")[:-1])


class RevisitedDataset(BaseDataset):

    HIERARCHY_LEVEL: int = 1

    def __init__(
        self,
        data_dir: str,
        mode: str = 'train',
        imsize: int = None,
        transform: Optional[Callable] = None,
        **kwargs: KwargsType,
    ) -> NoneType:
        assert mode in ["query", "gallery"]

        self.data_dir = lib.expand_path(data_dir)
        self.mode = mode
        self.imsize = imsize
        self.transform = transform
        self.city = self.data_dir.split('/')
        self.city = self.city[-1] if self.city[-1] else self.city[-2]

        with open(join(self.data_dir, f"gnd_{self.city}.pkl"), "rb") as f:
            db = pickle.load(f)

        self.paths = [join(self.data_dir, "jpg", f"{x}.jpg") for x in db["qimlist" if self.mode == "query" else "imlist"]]
        self.labels_name = [path_to_label(x) for x in self.paths]
        labels_name_to_id = {lb: i for i, lb in enumerate(sorted(set(self.labels_name)))}
        self.labels = [labels_name_to_id[x] for x in self.labels_name]

        if self.mode == "query":
            self.bbx = [x["bbx"] for x in db["gnd"]]
            self.easy = [x["easy"] for x in db["gnd"]]
            self.hard = [x["hard"] for x in db["gnd"]]
            self.junk = [x["junk"] for x in db["gnd"]]

        kwargs['compute_relevances'] = False
        super().__init__(**kwargs)

    def __getitem__(self, idx: int) -> Mapping[str, List[int]]:
        img = Image.open(self.paths[idx])
        imfullsize = max(img.size)

        if self.mode == 'query':
            img = img.crop(self.bbx[idx])

        if self.imsize is not None:
            if self.mode == 'query':
                img = imresize(img, self.imsize * max(img.size) / imfullsize)
            else:
                img = imresize(img, self.imsize)

        if self.transform is not None:
            img = self.transform(img)

        out = {"image": img, "label": torch.tensor([self.labels[idx]])}
        if self.mode == 'query':
            out["easy"] = self.easy[idx]
            out["hard"] = self.hard[idx]
            out["junk"] = self.junk[idx]

        return out

    def __repr__(self,) -> str:
        return f"{self.city.title()}Dataset(mode={self.mode}, imsize={self.imsize}, len={len(self)})"
