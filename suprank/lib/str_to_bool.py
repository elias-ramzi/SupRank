from typing import Union


def str_to_bool(condition: Union[str, int, bool]) -> bool:
    condition = str(condition)

    if condition.lower() in ['true', '1']:
        condition = True
    elif condition.lower() in ['false', '0']:
        condition = False

    return condition
