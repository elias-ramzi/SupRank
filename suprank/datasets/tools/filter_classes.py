import numpy as np


def filter_classes(labels: np.array, num_samples_th: int, filter_on_hierarchy_level: int = 0) -> np.ndarray:
    labels = labels[:, filter_on_hierarchy_level]
    unique_labels, index, count = np.unique(labels, return_inverse=True, return_counts=True)

    count = count >= num_samples_th
    accepted_index = count[index]

    return accepted_index
