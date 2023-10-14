import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Mapping

from ml.models.utils import *


# mean=0 std=1
def bern_like(x: Tensor, p: float) -> Tensor:
    return torch.bernoulli(torch.full_like(x, p)) * (1 / p) - 1


# mean=0 std=1
def uniform_like(x: Tensor) -> Tensor:
    return (torch.rand_like(x) - 0.5) * np.sqrt(12)


def rand_like(x: Tensor, dist_name: str, std: float) -> Tensor:
    if dist_name == "bernoulli":
        return bern_like(x, 1 / (1 + std * std))
    elif dist_name == "rademacher":
        return bern_like(x, 0.5) * std
    elif dist_name == "uniform":
        return uniform_like(x) * std
    elif dist_name == "normal":
        return torch.randn_like(x) * std
    else:
        raise NotImplementedError


dists: Mapping[str, Callable[[Tensor, float], Tensor]] = {
    "bernoulli": lambda x, std: bern_like(x, 1 / (1 + std * std)),
    "rademacher": lambda x, std: bern_like(x, 0.5) * std,
    "uniform": lambda x, std: uniform_like(x) * std,
    "normal": lambda x, std: torch.randn_like(x) * std,
}

means: Mapping[str, Callable[[Tensor], Tensor]] = {
    "element": lambda x: x,
    "sample": lambda x: x.mean(tuple(range(1, x.dim())), keepdim=True).expand_as(x),
    "feature": lambda x: x.mean(0, keepdim=True).expand_as(x),
    "batch": lambda x: x.mean(tuple(range(x.dim())), keepdim=True).expand_as(x),
}


class DropoutConfig:
    def __init__(self, data: dict, std) -> None:
        self.data = data
        self.std = std

    def __getitem__(self, i) -> "DropoutConfig":
        std = self.std if isinstance(self.std, float) else self.std[i]
        return DropoutConfig(self.data, std)


class Dropout(nn.Module):
    def __init__(self, config: DropoutConfig) -> None:
        super().__init__()
        self.std = config.std
        self.dist = config.data["dist"]
        self.mean = config.data["mean"]
        assert isinstance(self.std, float)
        assert self.dist in dists
        assert self.mean in means

    def extra_repr(self) -> str:
        return f"std={self.std}, dist={self.dist}, mean={self.mean}"

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.std != 0.0:
            d = dists[self.dist](x, self.std)
            r = means[self.mean](x.square()).sqrt()
            x = x + d * r
        return x
