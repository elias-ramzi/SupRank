from typing import Mapping, Any

from torch import Tensor

import suprank.lib as lib
from suprank.engine.metrics import hap
from suprank.engine.compute_relevance_on_the_fly import get_relevances_pop

KwargsType = Mapping[str, Any]


def hap_LM(
    sorted_target: Tensor,
    hierarchy_levels: int,
    alpha: float = 1.0,
    reduce: bool = True,
    **kwargs: KwargsType,
) -> Tensor:
    relevances = get_relevances_pop(
        sorted_target,
        alpha=alpha,
        levels=hierarchy_levels,
    )
    sorted_rel = lib.create_relevance_matrix(sorted_target, relevances)

    return hap(sorted_rel, relevances, reduce=reduce)
