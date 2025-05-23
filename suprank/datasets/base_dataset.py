from typing import Optional, Union, List, Mapping, Type
import os
from os.path import join, isfile

import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image

import suprank.lib as lib
import suprank.engine as eng

NoneType = Type[None]


class BaseDataset(Dataset):

    RELEVANCE_BS = 256
    CACHE_FILE = "HAPPIER_relevances_{mode}_{relevance_type}_{alpha}.trch"

    def __init__(
        self,
        restict_hierarchy: Optional[Union[int, List[int]]] = None,
        compute_relevances: bool = True,
        alpha: float = 1.0,
        alpha_train: Optional[float] = None,
        relevance_type: str = 'pop',
        relevance_type_train: Optional[str] = None,
        cache_relevances: bool = True,
        force_reload: bool = False,
    ) -> NoneType:
        super().__init__()
        self.restict_hierarchy = restict_hierarchy
        self.alpha = alpha
        self.alpha_train = alpha if alpha_train is None else alpha_train
        self.relevance_type = relevance_type
        self.relevance_type_train = relevance_type if relevance_type_train is None else relevance_type_train

        self.epoch = 0

        if compute_relevances and (self.mode != 'train'):
            cache_path = None
            if cache_relevances:
                cache_path = join(
                    self.data_dir,
                    self.CACHE_FILE.format(mode=self.mode, relevance_type=self.relevance_type, alpha=self.alpha)
                )

                if isfile(cache_path) and not force_reload:
                    self.relevances = torch.load(cache_path, map_location='cpu')
                else:
                    self.get_all_relevance(cache=cache_path, verbose=True)

            else:
                self.get_all_relevance(cache=None, verbose=True)
        else:
            self.relevances = None

    def __len__(self,) -> int:
        return len(self.paths)

    @property
    def my_sub_repr(self,) -> str:
        return ""

    def get_all_relevance(
        self,
        verbose: bool = False,
        cache: Optional[str] = None,
    ) -> NoneType:
        all_relevances = []
        BS = self.RELEVANCE_BS
        lib.LOGGER.info("Launching relevance computation")
        iterator = lib.track(
            range(len(self) // BS + (len(self) % BS != 0)),
            f"Creating relevance for {self.__class__.__name__} mode={self.mode} relevance_type={self.relevance_type} alpha={self.alpha}",
            disable=not (verbose and (not os.getenv('TQDM_DISABLE')))
        )
        torch_labels = torch.from_numpy(self.labels).to('cuda' if os.getenv("USE_CUDA_FOR_RELEVANCE") else 'cpu')
        for i in iterator:
            mask = torch.ones(len(torch_labels), dtype=torch.bool)
            mask[i * BS:(i + 1) * BS] = False
            target = lib.create_label_matrix(torch_labels[~mask], torch_labels)
            rel = eng.relevance_for_batch(
                target,
                alpha=self.alpha,
                check_for=range(self.HIERARCHY_LEVEL + 1),
                type=self.relevance_type,
            )
            all_relevances.append(rel)
        self.relevances = torch.cat(all_relevances).cpu()
        torch.cuda.empty_cache()

        if cache is not None:
            torch.save(self.relevances, cache)

    def compute_relevance_on_the_fly(self, target: Tensor, train: bool = True) -> Tensor:
        relevance_type = self.relevance_type_train if train else self.relevance_type
        alpha = self.alpha_train if train else self.alpha

        return eng.compute_relevance_on_the_fly(
            target,
            alpha=alpha,
            check_for=range(self.HIERARCHY_LEVEL + 1),
            type=relevance_type,
        )

    def set_epoch(self, e: int) -> NoneType:
        self.epoch = e

    def __getitem__(self, idx: int) -> Mapping[str, Union[Tensor, str, int]]:
        pth = self.paths[idx]
        img = Image.open(pth).convert('RGB')
        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx, :])
        out = {"image": img, "label": label, "path": pth, "index": idx}

        if self.relevances is not None:
            relevances = self.relevances[idx, :]
            out["relevance"] = relevances

        return out

    def __repr__(self) -> str:
        repr = (
            f"{self.__class__.__name__}(\n"
            f"    mode={self.mode},\n"
            f"    len={len(self)},\n"
            f"    L={self.HIERARCHY_LEVEL},\n"
            f"    restict_hierarchy={self.restict_hierarchy},\n"
        )

        if self.relevances is not None:
            repr = repr + f"    alpha={self.alpha}\n"
            repr = repr + f"    relevance_type={self.relevance_type}\n"

        repr = repr + self.my_sub_repr + ')'
        return repr
