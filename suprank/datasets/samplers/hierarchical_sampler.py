from typing import Optional, Iterator, List, Type
import math

import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_metric_learning.samplers import HierarchicalSampler as PMLHierarchicalSampler

import suprank.lib as lib

NoneType = Type[None]


class HierarchicalSampler(PMLHierarchicalSampler):

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 256,
        samples_per_class: int = 4,
        batches_per_super_tuple: int = 4,
        super_classes_per_batch: int = 2,
        inner_label: int = 0,
        outer_label: int = 1,
        restrict_number_of_batches: Optional[int] = None,
    ) -> NoneType:
        self.restrict_number_of_batches = restrict_number_of_batches
        self.epoch = 0

        super().__init__(
            dataset.labels,
            batch_size=batch_size,
            samples_per_class=samples_per_class,
            batches_per_super_tuple=batches_per_super_tuple,
            super_classes_per_batch=super_classes_per_batch,
            inner_label=inner_label,
            outer_label=outer_label,
        )

    def set_epoch(self, epoch: int) -> NoneType:
        self.epoch = epoch
        self.reshuffle()

    def __iter__(self,) -> Iterator[List[int]]:
        for batch in self.batches:
            yield batch

    def reshuffle(self) -> NoneType:
        lib.LOGGER.info("Shuffling data")
        super().reshuffle()
        if self.restrict_number_of_batches is not None:
            # batches are already shuffled
            self.batches = self.batches[:self.restrict_number_of_batches]

    def __repr__(self,) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"    batch_size={self.batch_size},\n"
            f"    samples_per_class={self.samples_per_class},\n"
            f"    batches_per_super_tuple={self.batches_per_super_tuple},\n"
            f"    super_classes_per_batch={self.super_classes_per_batch},\n"
            f"    restrict_number_of_batches={self.restrict_number_of_batches},\n"
            ")"
        )


class DistributedHierarchicalSampler(PMLHierarchicalSampler):

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 256,
        samples_per_class: int = 4,
        batches_per_super_tuple: int = 4,
        super_classes_per_batch: int = 2,
        inner_label: int = 0,
        outer_label: int = 1,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        split_batch: bool = False,
    ) -> NoneType:
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
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = int(shuffle)
        self.seed = seed + 1
        self.split_batch = split_batch

        self.num_samples = math.ceil(self.batch_size / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.padding_size = self.total_size - self.batch_size

        # For now forbid not shuffling
        assert shuffle == 1

        super().__init__(
            dataset.labels,
            batch_size=batch_size,
            samples_per_class=samples_per_class,
            batches_per_super_tuple=batches_per_super_tuple,
            super_classes_per_batch=super_classes_per_batch,
            inner_label=inner_label,
            outer_label=outer_label,
        )

        self.set_epoch(0)

    def __len__(self,) -> int:
        return len(self.worker_batches)

    def __iter__(self,) -> Iterator[List[int]]:
        for batch in self.worker_batches:
            yield batch

    def set_epoch(self, epoch: int) -> NoneType:
        self.epoch = epoch

        NP_STATE = np.random.get_state()
        np.random.seed((self.epoch + self.seed) * self.shuffle)
        super().reshuffle()
        if self.split_batch:
            # add extra samples to make it evenly divisible
            if self.padding_size > 1:
                self.batches = [batch + batch[:self.padding_size] for batch in self.batches]
            self.worker_batches = [batch[self.rank * self.num_samples: (self.rank + 1) * self.num_samples] for batch in self.batches]
        else:
            N = len(self.batches) // self.num_replicas
            self.worker_batches = self.batches[self.rank * N: (self.rank + 1) * N]
        np.random.set_state(NP_STATE)

    def __repr__(self,) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"    batch_size={self.batch_size},\n"
            f"    samples_per_class={self.samples_per_class},\n"
            f"    batches_per_super_tuple={self.batches_per_super_tuple},\n"
            f"    super_classes_per_batch={self.super_classes_per_batch},\n"
            ")"
        )


if __name__ == '__main__':

    from suprank.datasets.sop import SOPDataset

    dts = SOPDataset(
        '/local/DEEPLEARNING/image_retrieval/Stanford_Online_Products',
        mode='train',
    )

    np.random.seed(10)
    sampler = HierarchicalSampler(
        dts,
    )

    dist_sampler = DistributedHierarchicalSampler(
        dts,
        num_replicas=4,
        rank=0,
        split_batch=True,
    )
