from typing import Dict, Type
from collections import defaultdict

from suprank.lib.average_meter import AverageMeter

NoneType = Type[None]


class DictAverage(defaultdict):

    def __init__(self,) -> NoneType:
        super().__init__(AverageMeter)

    def update(self, dict_values: Dict[str, float], n: int = 1) -> NoneType:
        for key, item in dict_values.items():
            self[key].update(item, n)

    @property
    def avg(self,) -> Dict[str, float]:
        return {key: item.avg for key, item in self.items()}

    @property
    def sum(self,) -> Dict[str, float]:
        return {key: item.sum for key, item in self.items()}

    @property
    def count(self,) -> Dict[str, float]:
        return {key: item.count for key, item in self.items()}


if __name__ == '__main__':
    dict_avg = DictAverage()
    dict_avg.update({'a': 1, 'b': 2})
    dict_avg.update({'a': 2, 'b': 3})
    print(dict_avg.avg)
    print(dict_avg.sum)
    print(dict_avg.count)
