from typing import Optional, Iterator, List, Any, Type
import copy
import math

import numpy as np
import torch
from torch.utils.data import Dataset, BatchSampler
from pytorch_metric_learning import samplers

import suprank.lib as lib

NoneType = Type[None]


def flatten(list_: List[List[Any]]) -> List[Any]:
    return [item for sublist in list_ for item in sublist]


class MPerClassSampler(BatchSampler):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        samples_per_class: int = 4,
        hierarchy_level: int = 0,
        reduce_per_class: Optional[int] = None,
    ) -> NoneType:
        assert samples_per_class > 1
        assert batch_size % samples_per_class == 0

        labels = dataset.labels[:, hierarchy_level]
        self.image_dict = {cl: [] for cl in set(labels)}
        for idx, cl in enumerate(labels):
            self.image_dict[cl].append(idx)

        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.epoch = 0
        self.reduce_per_class = reduce_per_class

        self.reshuffle()

    def __iter__(self,) -> Iterator[List[int]]:
        for batch in self.batches:
            yield batch

    def __len__(self,) -> int:
        return len(self.batches)

    def __repr__(self,) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"    batch_size={self.batch_size},\n"
            f"    samples_per_class={self.samples_per_class}\n)"
        )

    def set_epoch(self, epoch: int) -> NoneType:
        self.epoch = epoch
        self.reshuffle()

    def reshuffle(self) -> NoneType:
        lib.LOGGER.info("Shuffling data")
        image_dict = copy.deepcopy(self.image_dict)
        for sub in image_dict:
            np.random.shuffle(image_dict[sub])
            if self.reduce_per_class is not None:
                image_dict[sub] = image_dict[sub][: self.reduce_per_class]

        classes = [*image_dict]
        np.random.shuffle(classes)
        total_batches = []
        batch = []
        finished = 0
        while finished == 0:
            for sub_class in classes:
                if (len(image_dict[sub_class]) >= self.samples_per_class) and (len(batch) < self.batch_size / self.samples_per_class):
                    batch.append(image_dict[sub_class][:self.samples_per_class])
                    image_dict[sub_class] = image_dict[sub_class][self.samples_per_class:]

            if len(batch) == self.batch_size / self.samples_per_class:
                batch = flatten(batch)
                np.random.shuffle(batch)
                total_batches.append(batch)
                batch = []
            else:
                finished = 1

        np.random.shuffle(total_batches)
        self.batches = total_batches


class PMLMPerClassSampler(samplers.MPerClassSampler):
    """
    This sampler is almost the same as the one above (the sampling
    method is not exactly the same). But it allows to choose
    how many batches are sampled.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_iter: int,
        samples_per_class: int = 4,
        hierarchy_level: int = 0,
    ) -> NoneType:
        super().__init__(
            dataset.labels[:, hierarchy_level],
            m=samples_per_class,
            batch_size=batch_size,
            length_before_new_iter=num_iter * batch_size,
        )

        self.samples_per_class = samples_per_class
        self.num_iter = num_iter
        self.epoch = 0
        self.reshuffle()

    def __iter__(self,) -> Iterator[List[int]]:
        for batch in self.batches:
            yield batch

    def __len__(self,) -> int:
        return len(self.batches)

    def __repr__(self,) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"    batch_size={self.batch_size},\n"
            f"    samples_per_class={self.samples_per_class},\n"
            f"    num_iter={self.num_iter},\n"
            ")"
        )

    def set_epoch(self, epoch: int) -> NoneType:
        self.epoch = epoch
        self.reshuffle()

    def reshuffle(self,) -> NoneType:
        lib.LOGGER.info("Shuffling data")
        idxs = list(super().__iter__())
        n = len(idxs)
        batches = []
        for i in range(n // self.batch_size):
            batches.append(idxs[:self.batch_size])
            del idxs[:self.batch_size]

        np.random.shuffle(batches)
        self.batches = batches


class DistributedMPerClassSampler(samplers.MPerClassSampler):

    def __init__(
        self,
        dataset: Dataset,
        num_iter: int = None,
        batch_size: int = 256,
        samples_per_class: int = 4,
        split_batch: bool = False,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ) -> NoneType:
        """
        labels: np.ndarray, labels of the dataset
        num_iter: number of iteration, default: None --> this gives the same number of iterations as with DistributedSampler
        batch_size: batch size per worker
        samples_per_class: number of items with the same class in each batch
        split_batch: if True a batch is splitter across workers, if False each workers gets a batch constructed as MPerClassSampler
        num_replicas: number of workers, better left to None
        rank: rank of the worker, better left to None
        seed: seed used for random generation
        """
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.num_iter = num_iter
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed + 1
        self.split_batch = split_batch
        self.samples_per_class = samples_per_class

        self.original_labels = dataset.labels

        self.num_samples = math.ceil(len(dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        if self.num_iter is None:
            # this gives the same number of iteration as for the vanilla DistributedSampler
            self.num_iter = self.total_size // (self.batch_size * self.num_replicas)

        self.effective_batch_size = self.batch_size * self.num_replicas if self.split_batch else self.batch_size
        self.effective_num_iter = self.num_iter if self.split_batch else self.num_iter * self.num_replicas

        super().__init__(
            dataset.labels,
            m=samples_per_class,
            batch_size=self.effective_batch_size,
            length_before_new_iter=self.effective_num_iter * self.effective_batch_size,
        )

    def __len__(self,) -> int:
        return (self.effective_num_iter * self.effective_batch_size) // self.num_replicas

    def __iter__(self,) -> Iterator:
        # print(self.original_labels[:10])
        NP_STATE = np.random.get_state()
        np.random.seed(self.epoch + self.seed)
        indices = list(super().__iter__())
        np.random.set_state(NP_STATE)

        assert len(indices) % self.effective_batch_size == 0
        batches = [indices[x:x + self.effective_batch_size] for x in range(0, len(indices), self.effective_batch_size)]

        if self.split_batch:
            raise NotImplementedError
            batches = [batch[self.rank * self.batch_size: (self.rank + 1) * self.batch_size] for batch in batches]
        else:
            N = len(batches) // self.num_replicas
            batches = batches[self.rank * N: (self.rank + 1) * N]

        indices = flatten(batches)
        return iter(indices)

    def __repr__(self,) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"    batch_size={self.batch_size},\n"
            f"    samples_per_class={self.samples_per_class},\n"
            ")"
        )

    def set_epoch(self, epoch: int, labels: Optional[torch.Tensor] = None) -> NoneType:
        self.epoch = epoch
