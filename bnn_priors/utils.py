import math
from functools import reduce
from typing import Callable

from torch import nn


def find_module_names(module: nn.Module, filter: Callable[[nn.Module], bool]):
    found_names = []
    for name, module in module.named_modules():
        if filter(module):
            found_names.append(name)
    return found_names


def get_module_by_name(module: nn.Module, name: str):
    names = name.split(sep='.')
    return reduce(getattr, names, module)


def set_module_by_name(module: nn.Module, name: str, replacement: nn.Module):
    names = name.split(sep='.')
    parent = reduce(getattr, names[:-1], module)
    setattr(parent, names[-1], replacement)


def get_cosine_schedule(samples_per_cycle: int) -> Callable[[int], float]:
    def schedule(i: int) -> float:
        cycle_progress = (i % samples_per_cycle) / samples_per_cycle
        scale = 0.5 * (math.cos(math.pi * cycle_progress) + 1.)
        return scale

    return schedule
