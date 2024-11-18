from typing import Mapping, Any, Optional, Type


def cast_dict_to_type(dct: Mapping[str, Any], keys_type: Optional[Type] = None, values_type: Optional[Type] = None) -> Mapping[Any, Any]:
    assert isinstance(dct, dict)
    assert (keys_type is not None) or (values_type is not None)

    if keys_type is None:
        keys_type = lambda x: x  # noqa E731

    if values_type is None:
        values_type = lambda x: x  # noqa E731

    return {keys_type(k): values_type(v) for k, v in dct.items()}
