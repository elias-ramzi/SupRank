"""
Created with the help Github Copilot
"""
from typing import Optional

import numpy as np
from torch.utils.data import Dataset, Subset


# Function that given a dataset will sub-sample it to a given number of images per class
# the classes are kept in the attribute self.labels as a numpy array
def sub_sample(
        dataset: Dataset,
        factor: Optional[float] = None,
        n_images_per_class: Optional[int] = None,
        random_state: int = 42,
) -> Dataset:
    assert (factor is not None) or (n_images_per_class is not None), "You must specify one of the two parameters 'factor' and 'n_images_per_class'."
    assert (factor is None) or (n_images_per_class is None), "You can only specify one of the two parameters 'factor' and 'n_images_per_class'."
    # get the labels of the dataset
    labels = dataset.labels[:, 0]
    # get the unique labels
    unique_labels = np.unique(labels)
    # create a list of indices that will be used to sub-sample the dataset
    indices = []
    # for each unique label
    for i, label in enumerate(unique_labels):
        # get the indices of the images with that label
        label_indices = np.where(labels == label)[0]
        # get the number of images with that label
        n_images = len(label_indices)
        # get the number of images that will be kept
        if factor is not None:
            n_images_kept = max(1, int(n_images * factor))
        elif n_images_per_class is not None:
            n_images_kept = min(n_images, n_images_per_class)
        # get the indices of the images that will be kept
        np.random.seed(random_state + i)
        label_indices_kept = np.random.choice(label_indices, n_images_kept, replace=False)
        # add the indices to the list of indices
        indices.extend(label_indices_kept)
    # create a new dataset with the sub-sampled images
    sub_sampled_dataset = Subset(dataset, indices)
    # set the labels of the new dataset
    sub_sampled_dataset.labels = dataset.labels[indices]

    if hasattr(dataset, "super_labels"):
        sub_sampled_dataset.super_labels = dataset.super_labels[indices]

    # return the new dataset
    return sub_sampled_dataset
