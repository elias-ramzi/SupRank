from typing import Mapping, Any, Optional, Callable, Type
import os

import numpy as np
from torchvision import datasets

import suprank.lib as lib
from suprank.datasets.base_dataset import BaseDataset

NoneType = Type[None]
KwargsType = Mapping[str, Any]


class Cub200Dataset(BaseDataset):

    HIERARCHY_LEVEL = 3

    def __init__(
        self,
        data_dir: str,
        mode: str = 'train',
        transform: Optional[Callable] = None,
        classification: bool = False,
        **kwargs: KwargsType,
    ) -> NoneType:
        self.data_dir = lib.expand_path(data_dir)
        self.mode = mode
        self.transform = transform
        self.classification = classification

        if self.classification:
            with open(os.path.join(self.data_dir, 'train_test_split.txt')) as f:
                splits = f.read().split("\n")
                ids = []
                for line in splits:
                    id_, is_train = line.split(" ")
                    if mode == 'train' and bool(line):
                        ids.append(id_)
                    elif mode == 'test' and not bool(line):
                        ids.append(id_)
                    elif mode == 'all':
                        ids.append(id_)
            with open(os.path.join(self.data_dir, 'images.txt')) as f:
                paths = f.read().split("\n")
                paths.remove("")
                paths = {x.split(" ")[0]: x.split(" ")[1] for x in paths}
                self.paths = [paths[x] for x in ids]
            with open(os.path.join(self.data_dir, 'image_class_labels.txt')) as f:
                labels = f.read().split("\n")
                labels = {x.split(" ")[0]: x.split(" ")[1] for x in labels}
                self.labels = []
                for x in ids:
                    lb = labels[x]
                    self.labels.append([trees[lb + 1][x] for x in [0, 2, 1]])
        else:
            dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'images'))
            paths = [a for (a, b) in dataset.imgs]
            labels = [b for (a, b) in dataset.imgs]

            sorted_lb = list(sorted(set(labels)))
            if mode == 'train':
                set_labels = set(sorted_lb[:len(sorted_lb) // 2])
            elif mode == 'test':
                set_labels = set(sorted_lb[len(sorted_lb) // 2:])
            elif mode == 'all':
                set_labels = sorted_lb

            self.paths = []
            self.labels = []
            for lb, pth in zip(labels, paths):
                if lb in set_labels:
                    self.paths.append(pth)
                    self.labels.append([trees[lb + 1][x] for x in [0, 2, 1]])

        self.labels = np.array(self.labels)
        self.labels = lib.set_labels_to_range(self.labels)
        super().__init__(**kwargs)


trees = {
    1: [1, 12, 35],
    2: [2, 12, 35],
    3: [3, 12, 35],
    4: [4, 6, 9],
    5: [5, 4, 4],
    6: [6, 4, 4],
    7: [7, 4, 4],
    8: [8, 4, 4],
    9: [9, 8, 18],
    10: [10, 8, 18],
    11: [11, 8, 18],
    12: [12, 8, 18],
    13: [13, 8, 18],
    14: [14, 8, 13],
    15: [15, 8, 13],
    16: [16, 8, 13],
    17: [17, 8, 13],
    18: [18, 8, 26],
    19: [19, 8, 21],
    20: [20, 8, 19],
    21: [21, 8, 24],
    22: [22, 3, 3],
    23: [23, 13, 37],
    24: [24, 13, 37],
    25: [25, 13, 37],
    26: [26, 8, 18],
    27: [27, 8, 18],
    28: [28, 8, 14],
    29: [29, 8, 15],
    30: [30, 8, 15],
    31: [31, 6, 9],
    32: [32, 6, 9],
    33: [33, 6, 9],
    34: [34, 8, 16],
    35: [35, 8, 16],
    36: [36, 10, 33],
    37: [37, 8, 30],
    38: [38, 8, 30],
    39: [39, 8, 30],
    40: [40, 8, 30],
    41: [41, 8, 30],
    42: [42, 8, 30],
    43: [43, 8, 30],
    44: [44, 13, 38],
    45: [45, 12, 36],
    46: [46, 1, 1],
    47: [47, 8, 16],
    48: [48, 8, 16],
    49: [49, 8, 18],
    50: [50, 11, 34],
    51: [51, 11, 34],
    52: [52, 11, 34],
    53: [53, 11, 34],
    54: [54, 8, 13],
    55: [55, 8, 16],
    56: [56, 8, 16],
    57: [57, 8, 13],
    58: [58, 4, 4],
    59: [59, 4, 5],
    60: [60, 4, 5],
    61: [61, 4, 5],
    62: [62, 4, 5],
    63: [63, 4, 5],
    64: [64, 4, 5],
    65: [65, 4, 5],
    66: [66, 4, 5],
    67: [67, 2, 2],
    68: [68, 2, 2],
    69: [69, 2, 2],
    70: [70, 2, 2],
    71: [71, 4, 6],
    72: [72, 4, 6],
    73: [73, 8, 15],
    74: [74, 8, 15],
    75: [75, 8, 15],
    76: [76, 8, 24],
    77: [77, 8, 30],
    78: [78, 8, 30],
    79: [79, 5, 7],
    80: [80, 5, 7],
    81: [81, 5, 7],
    82: [82, 5, 7],
    83: [83, 5, 7],
    84: [84, 5, 8],
    85: [85, 8, 11],
    86: [86, 7, 10],
    87: [87, 1, 1],
    88: [88, 8, 18],
    89: [89, 1, 1],
    90: [90, 1, 1],
    91: [91, 8, 21],
    92: [92, 3, 3],
    93: [93, 8, 15],
    94: [94, 8, 27],
    95: [95, 8, 18],
    96: [96, 8, 18],
    97: [97, 8, 18],
    98: [98, 8, 18],
    99: [99, 8, 23],
    100: [100, 9, 32],
    101: [101, 9, 32],
    102: [102, 8, 30],
    103: [103, 8, 30],
    104: [104, 8, 22],
    105: [105, 3, 3],
    106: [106, 4, 4],
    107: [107, 8, 15],
    108: [108, 8, 15],
    109: [109, 8, 23],
    110: [110, 6, 9],
    111: [111, 8, 20],
    112: [112, 8, 20],
    113: [113, 8, 24],
    114: [114, 8, 24],
    115: [115, 8, 24],
    116: [116, 8, 24],
    117: [117, 8, 24],
    118: [118, 8, 25],
    119: [119, 8, 24],
    120: [120, 8, 24],
    121: [121, 8, 24],
    122: [122, 8, 24],
    123: [123, 8, 24],
    124: [124, 8, 24],
    125: [125, 8, 24],
    126: [126, 8, 24],
    127: [127, 8, 24],
    128: [128, 8, 24],
    129: [129, 8, 24],
    130: [130, 8, 24],
    131: [131, 8, 24],
    132: [132, 8, 24],
    133: [133, 8, 24],
    134: [134, 8, 28],
    135: [135, 8, 17],
    136: [136, 8, 17],
    137: [137, 8, 17],
    138: [138, 8, 17],
    139: [139, 8, 13],
    140: [140, 8, 13],
    141: [141, 4, 5],
    142: [142, 4, 5],
    143: [143, 4, 5],
    144: [144, 4, 5],
    145: [145, 4, 5],
    146: [146, 4, 5],
    147: [147, 4, 5],
    148: [148, 8, 24],
    149: [149, 8, 21],
    150: [150, 8, 21],
    151: [151, 8, 31],
    152: [152, 8, 31],
    153: [153, 8, 31],
    154: [154, 8, 31],
    155: [155, 8, 31],
    156: [156, 8, 31],
    157: [157, 8, 31],
    158: [158, 8, 23],
    159: [159, 8, 23],
    160: [160, 8, 23],
    161: [161, 8, 23],
    162: [162, 8, 23],
    163: [163, 8, 23],
    164: [164, 8, 23],
    165: [165, 8, 23],
    166: [166, 8, 23],
    167: [167, 8, 23],
    168: [168, 8, 23],
    169: [169, 8, 23],
    170: [170, 8, 23],
    171: [171, 8, 23],
    172: [172, 8, 23],
    173: [173, 8, 23],
    174: [174, 8, 23],
    175: [175, 8, 23],
    176: [176, 8, 23],
    177: [177, 8, 23],
    178: [178, 8, 23],
    179: [179, 8, 23],
    180: [180, 8, 23],
    181: [181, 8, 23],
    182: [182, 8, 23],
    183: [183, 8, 23],
    184: [184, 8, 23],
    185: [185, 8, 12],
    186: [186, 8, 12],
    187: [187, 10, 33],
    188: [188, 10, 33],
    189: [189, 10, 33],
    190: [190, 10, 33],
    191: [191, 10, 33],
    192: [192, 10, 33],
    193: [193, 8, 29],
    194: [194, 8, 29],
    195: [195, 8, 29],
    196: [196, 8, 29],
    197: [197, 8, 29],
    198: [198, 8, 29],
    199: [199, 8, 29],
    200: [200, 8, 23]
}


if __name__ == '__main__':
    dts = Cub200Dataset("~/datasets/CUB_200_2011", "train")
