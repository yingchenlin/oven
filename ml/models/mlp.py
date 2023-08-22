import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Mapping

from ml.datasets import Dataset
from ml.models.utils import *


def get_activation(config: dict) -> nn.Module:
    name = config["name"]
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise NotImplementedError(name)


# mean=0 std=1
def bern_like(x: Tensor, p: float):
    return torch.bernoulli(torch.full_like(x, p)) * (1 / p) - 1


# mean=0 std=1
def uniform_like(x: Tensor):
    return (torch.rand_like(x) - 0.5) * np.sqrt(12)


dists: Mapping[str, Callable[[Tensor, float], Tensor]] = {
    "bernoulli": lambda x, std: bern_like(x, 1 / (1 + std * std)),
    "rademacher": lambda x, std: bern_like(x, 0.5) * std,
    "uniform": lambda x, std: uniform_like(x) * std,
    "normal": lambda x, std: torch.randn_like(x) * std,
}

means: Mapping[str, Callable[[Tensor], Tensor]] = {
    # general: [sample, batch, feature...]
    "element": lambda x: x,
    "sample": lambda x: x.mean(tuple(range(2, x.dim())), keepdim=True).expand_as(x),
    "feature": lambda x: x.mean([0, 1], keepdim=True).expand_as(x),
    "batch": lambda x: x.mean(tuple(range(x.dim())), keepdim=True).expand_as(x),
    # cnn: [sample, batch, channel, height, width]
    "sample-pixel-2d": lambda x: x.mean(2, keepdim=True).expand_as(x),
    "sample-channel-2d": lambda x: x.mean([3, 4], keepdim=True).expand_as(x),
    "pixel-2d": lambda x: x.mean([0, 1, 2], keepdim=True).expand_as(x),
    "channel-2d": lambda x: x.mean([0, 1, 3, 4], keepdim=True).expand_as(x),
}


class Dropout(nn.Module):
    def __init__(self, config: dict, layer: int) -> None:
        super().__init__()
        assert config["dist"] in dists
        assert config["mean"] in means
        self.std = config["std"][layer]
        self.dist = config["dist"]
        self.mean = config["mean"]

    def extra_repr(self) -> str:
        return f"std={self.std}, dist={self.dist}, mean={self.mean}"

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.std != 0.0:
            d = dists[self.dist](x, self.std)
            r = means[self.mean](x.square()).sqrt()
            x = x + d * r
        return x


class MLP(nn.Sequential, Model):
    _flatten = nn.Flatten
    _dropout = Dropout
    _linear = nn.Linear
    _activation = lambda _, config: get_activation(config)

    def __init__(self, config: dict, dataset: Dataset) -> None:
        num_layers: int = config["num_layers"]
        hidden_dim: int = config["hidden_dim"]
        output_dim: int = int(np.prod(dataset.input_shape))

        layers = []
        layers.append(self._flatten(start_dim=-len(dataset.input_shape)))
        for i in range(num_layers):
            input_dim, output_dim = output_dim, hidden_dim
            layers.append(self._dropout(config["dropout"], i))
            layers.append(self._linear(input_dim, output_dim))
            layers.append(self._activation(config["activation"]))
            layers.append(Capture())
        input_dim, output_dim = output_dim, dataset.num_classes
        layers.append(self._dropout(config["dropout"], num_layers))
        layers.append(self._linear(input_dim, output_dim))
        layers.append(Capture())

        super().__init__(*layers)


class Regulator(Dropout):
    input: Tensor

    def forward(self, x: Tensor) -> Tensor:
        self.input = x
        return x


class RegMLP(MLP):
    _dropout = Regulator

    def __init__(self, config: dict, dataset: Dataset) -> None:
        super().__init__(config, dataset)
        self.diff = config["dropout"]["diff"]

    def reg_loss(self, outputs: Tensor) -> Tensor:
        losses = torch.zeros(())
        if self.training:
            prob = outputs.softmax(-1)
            core = (prob.diag_embed() + outer(prob) * (self.diff - 1)) * 0.5
            jacob = torch.ones_like(prob).diag_embed()
            for m in reversed(self):
                if isinstance(m, Regulator):
                    jacob = jacob @ m.weight
                    cov = None
                    if m.training and m.std != 0.0:
                        var = means[m.mean](m.input.square()) * m.std**2
                        cov = torch.einsum("sbik,sbjk,sbk->sbij", jacob, jacob, var)
                    jacob = jacob * (m.input > 0).unsqueeze(-2)
                    if cov is not None:
                        losses = losses + torch.einsum("sbij,sbij->sb", core, cov)
        return losses
