from typing import Optional, Callable, Mapping, Any, Type
from os.path import join

import numpy as np
from scipy.io import loadmat

import suprank.lib as lib
from suprank.datasets.base_dataset import BaseDataset

NoneType = Type[None]
KwargsType = Mapping[str, Any]


class Cars196Dataset(BaseDataset):

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

        img_data = loadmat(join(self.data_dir, "cars_annos.mat"))
        labels = np.array([i[0, 0] for i in img_data["annotations"]["class"][0]])
        paths = [join(self.data_dir, i[0]) for i in img_data["annotations"]["relative_im_path"][0]]

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
                self.labels.append(trees[lb])

        self.labels = np.array(self.labels)
        self.labels = lib.set_labels_to_range(self.labels)
        super().__init__(**kwargs)


trees = {
    1: [1, 7],
    2: [2, 6],
    3: [3, 6],
    4: [4, 6],
    5: [5, 6],
    6: [6, 3],
    7: [7, 4],
    8: [8, 2],
    9: [9, 3],
    10: [10, 2],
    11: [11, 3],
    12: [12, 2],
    13: [13, 3],
    14: [14, 3],
    15: [15, 3],
    16: [16, 6],
    17: [17, 6],
    18: [18, 9],
    19: [19, 4],
    20: [20, 6],
    21: [21, 2],
    22: [22, 3],
    23: [23, 6],
    24: [24, 6],
    25: [25, 3],
    26: [26, 6],
    27: [27, 2],
    28: [28, 3],
    29: [29, 6],
    30: [30, 9],
    31: [31, 2],
    32: [32, 7],
    33: [33, 7],
    34: [34, 3],
    35: [35, 6],
    36: [36, 2],
    37: [37, 7],
    38: [38, 2],
    39: [39, 2],
    40: [40, 6],
    41: [41, 6],
    42: [42, 3],
    43: [43, 3],
    44: [44, 6],
    45: [45, 2],
    46: [46, 3],
    47: [47, 6],
    48: [48, 7],
    49: [49, 6],
    50: [50, 7],
    51: [51, 6],
    52: [52, 7],
    53: [53, 1],
    54: [54, 1],
    55: [55, 2],
    56: [56, 3],
    57: [57, 3],
    58: [58, 7],
    59: [59, 2],
    60: [60, 5],
    61: [61, 6],
    62: [62, 7],
    63: [63, 6],
    64: [64, 8],
    65: [65, 1],
    66: [66, 3],
    67: [67, 6],
    68: [68, 7],
    69: [69, 1],
    70: [70, 1],
    71: [71, 8],
    72: [72, 3],
    73: [73, 6],
    74: [74, 1],
    75: [75, 1],
    76: [76, 7],
    77: [77, 2],
    78: [78, 5],
    79: [79, 6],
    80: [80, 2],
    81: [81, 2],
    82: [82, 9],
    83: [83, 9],
    84: [84, 9],
    85: [85, 5],
    86: [86, 1],
    87: [87, 1],
    88: [88, 8],
    89: [89, 7],
    90: [90, 1],
    91: [91, 1],
    92: [92, 9],
    93: [93, 3],
    94: [94, 7],
    95: [95, 7],
    96: [96, 6],
    97: [97, 6],
    98: [98, 4],
    99: [99, 3],
    100: [100, 2],
    101: [101, 3],
    102: [102, 2],
    103: [103, 2],
    104: [104, 3],
    105: [105, 6],
    106: [106, 1],
    107: [107, 2],
    108: [108, 5],
    109: [109, 7],
    110: [110, 7],
    111: [111, 1],
    112: [112, 3],
    113: [113, 1],
    114: [114, 1],
    115: [115, 6],
    116: [116, 9],
    117: [117, 6],
    118: [118, 7],
    119: [119, 8],
    120: [120, 7],
    121: [121, 7],
    122: [122, 1],
    123: [123, 2],
    124: [124, 1],
    125: [125, 1],
    126: [126, 5],
    127: [127, 5],
    128: [128, 3],
    129: [129, 6],
    130: [130, 4],
    131: [131, 7],
    132: [132, 7],
    133: [133, 7],
    134: [134, 6],
    135: [135, 6],
    136: [136, 6],
    137: [137, 6],
    138: [138, 6],
    139: [139, 4],
    140: [140, 6],
    141: [141, 3],
    142: [142, 7],
    143: [143, 7],
    144: [144, 3],
    145: [145, 7],
    146: [146, 7],
    147: [147, 7],
    148: [148, 7],
    149: [149, 7],
    150: [150, 3],
    151: [151, 3],
    152: [152, 3],
    153: [153, 3],
    154: [154, 7],
    155: [155, 7],
    156: [156, 6],
    157: [157, 2],
    158: [158, 2],
    159: [159, 7],
    160: [160, 3],
    161: [161, 2],
    162: [162, 6],
    163: [163, 3],
    164: [164, 6],
    165: [165, 6],
    166: [166, 8],
    167: [167, 6],
    168: [168, 4],
    169: [169, 8],
    170: [170, 4],
    171: [171, 3],
    172: [172, 3],
    173: [173, 6],
    174: [174, 5],
    175: [175, 2],
    176: [176, 6],
    177: [177, 6],
    178: [178, 4],
    179: [179, 2],
    180: [180, 3],
    181: [181, 6],
    182: [182, 6],
    183: [183, 4],
    184: [184, 6],
    185: [185, 6],
    186: [186, 7],
    187: [187, 6],
    188: [188, 6],
    189: [189, 7],
    190: [190, 4],
    191: [191, 4],
    192: [192, 4],
    193: [193, 4],
    194: [194, 6],
    195: [195, 7],
    196: [196, 2],
}


if __name__ == '__main__':
    dts = Cars196Dataset("~/datasets/cars196", "train")
