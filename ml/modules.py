from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional

from ml.datasets import Dataset


def get_model(config: dict, dataset: Dataset) -> MLP:
    name = config["name"]
    if name == "mlp":
        return MLP(config, dataset)
    if name == "dist-mlp":
        return DistMLP(config, dataset)
    raise NotImplementedError


def get_optim(config: dict, model: nn.Module) -> torch.optim.Optimizer:
    name = config["name"]
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config["learning_rate"],
            momentum=config["momentum"],
        )
    raise NotImplementedError


def get_loss_fn(config: dict) -> nn.Module:
    name = config["name"]
    if name == "ce":
        return CrossEntropyLoss(config)
    if name == "dist-ce":
        return DistCrossEntropyLoss(config)
    raise NotImplementedError


def get_activation(config: dict) -> nn.Module:
    name = config["name"]
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise NotImplementedError


class Capture(nn.Module):
    state: Optional[Tensor]

    def forward(self, input: Tensor) -> Tensor:
        self.state = input
        if isinstance(self.state, tuple):
            self.state = self.state[0]
        if self.state.requires_grad:
            self.state.retain_grad()
        return input


# utils


def outer(x: Tensor) -> Tensor:
    return x.unsqueeze(-1) * x.unsqueeze(-2)


def diag(x: Tensor) -> Tensor:
    return x.diagonal(0, -2, -1)


def gaussian(z: Tensor) -> Tuple[Tensor, Tensor]:
    g0 = (z.square() * -0.5).exp() * np.sqrt(1 / (np.pi * 2))
    g1 = ((z * np.sqrt(0.5)).erf() + 1) * 0.5
    return g0, g1


def cross_entropy(x: Tensor, i: Tensor) -> Tensor:
    return x.logsumexp(-1) - x.gather(-1, i.unsqueeze(-1)).squeeze(-1)


# MLP


# mean=0 std=1
def st_bern_like(x: Tensor, odds: float):
    return torch.bernoulli(torch.full_like(x, 1 / odds)) * odds - 1


# mean=0 std=1
def st_unif_like(x: Tensor):
    return (torch.rand_like(x) - 0.5) * np.sqrt(12)


rand_likes = {
    "bernoulli": lambda x, std: st_bern_like(x, 1 + std * std),
    "rademacher": lambda x, std: st_bern_like(x, 2) * std,
    "uniform": lambda x, std: st_unif_like(x) * std,
    "normal": lambda x, std: torch.randn_like(x) * std,
}


class Dropout(nn.Linear):
    input: Tensor

    def __init__(
        self, config: dict, layer: int, input_dim: int, output_dim: int
    ) -> None:
        super().__init__(input_dim, output_dim)
        self.std = config["std"] if layer in config["layers"] else 0
        self.dist = config["dist"]
        self.scheme = config["scheme"]
        self.scale = config["scale"]
        self.target = config["target"]
        assert self.scheme in ("dropout", "reg_loss")

    def extra_repr(self) -> str:
        m = {
            "std": self.std,
            "dist": self.dist,
            "scheme": self.scheme,
            "scale": self.scale,
            "target": self.target,
        }
        return " ".join([f"{k}={v}" for k, v in m.items()])

    def _perturbate(self, x: Tensor) -> Tensor:
        if self.scheme == "dropout":
            d = rand_likes[self.dist](x, self.std)
            return x + d * self._mean(x.square()).sqrt()
        return x

    def _mean(self, x: Tensor) -> Tensor:
        if self.scale == "element":
            return x
        elif self.scale == "sample":
            return x.mean(-1, keepdim=True).expand_as(x)
        elif self.scale == "feature":
            return x.mean(-2, keepdim=True).expand_as(x)
        elif self.scale == "batch":
            return x.mean([-2, -1], keepdim=True).expand_as(x)
        else:
            raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        self.input = x
        w = self.weight
        b = self.bias
        if self.training and self.std != 0.0:
            if self.target == "input":
                x = self._perturbate(x)
            elif self.target == "weight":
                w = self._perturbate(w)
            else:
                raise NotImplementedError
        return F.linear(x, w, b)

    def reg_loss(self, prob: Tensor, jacob: Tensor) -> Tuple[Tensor, Tensor]:
        loss = torch.zeros(())
        jacob = jacob @ self.weight
        hess = prob.diag_embed() - outer(prob)
        if self.training and self.std != 0.0:
            var = self._mean(self.input.square())
            cov = torch.einsum("sbik,sbjk,sbk->sbij", jacob, jacob, var)
            loss = torch.einsum("sbij,sbij->sb", hess, cov) * (self.std**2 / 2)
        jacob = jacob * (self.input > 0).unsqueeze(-2)
        return loss, jacob


class MLP(nn.Sequential):
    _flatten = nn.Flatten
    _activation = lambda _, config: get_activation(config)
    _dropout = Dropout

    def __init__(self, config: dict, dataset: Dataset) -> None:
        num_layers: int = config["num_layers"]
        hidden_dim: int = config["hidden_dim"]
        output_dim: int = int(np.prod(dataset.input_shape))

        layers = []
        layers.append(self._flatten(start_dim=-3))
        for i in range(num_layers):
            input_dim, output_dim = output_dim, hidden_dim
            layers.append(self._dropout(config["dropout"], i, input_dim, output_dim))
            layers.append(self._activation(config["activation"]))
            layers.append(Capture())
        input_dim, output_dim = output_dim, dataset.num_classes
        layers.append(
            self._dropout(config["dropout"], num_layers, input_dim, output_dim)
        )
        layers.append(Capture())

        super().__init__(*layers)

    def reg_loss(self, outputs: Tensor) -> Tensor:
        losses = torch.zeros(())
        if self.training:
            dropouts = [m for m in self if isinstance(m, Dropout)]
            if dropouts[0].scheme == "reg_loss":
                prob = outputs.softmax(-1)
                jacob = torch.ones_like(prob).diag_embed()
                for m in reversed(dropouts):
                    loss, jacob = m.reg_loss(prob, jacob)
                    losses = losses + loss
        return losses


class CrossEntropyLoss(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.diff = config["diff"]

    def forward(self, outputs, targets):
        return cross_entropy(outputs, targets)


"""
class DiffLoss(Loss):
    pass


class DropoutNormCrossEntropyLoss(DiffLoss):
    def __init__(self, config):
        super().__init__(config)
        self.kind = config["kind"]

    def forward(self, outputs, targets):
        mean, diff = outputs

        if self.kind == "prob":
            p = mean.softmax(-1)
            return (
                cross_entropy(mean, targets)
                + ((diff.square() * p).sum(-1) - (diff * p).sum(-1).square()) * 0.5
            )
        elif self.kind == "mean":
            return cross_entropy(mean, targets) + diff.square().mean(-1) * 0.5
        elif self.kind == "even":
            return (
                cross_entropy(mean + diff, targets)
                + cross_entropy(mean - diff, targets)
            ) * 0.5

        raise NotImplementedError()
"""

# DistMLP

DistTensor = Tuple[Tensor, Optional[Tensor]]


def get_dist_activation(config: dict) -> nn.Module:
    name = config["name"]
    if name == "relu":
        return DistReLU()
    raise NotImplementedError


class DistFlatten(nn.Flatten):
    def forward(self, input: Tensor) -> DistTensor:
        return super().forward(input), None


class DistReLU(nn.Module):
    def forward(self, input: DistTensor) -> DistTensor:
        m, k = input
        if k is None:
            return F.relu(m), None
        s = diag(k).sqrt() + 1e-8
        g0, g1 = gaussian(m / s)
        m = m * g1 + s * g0
        k = k * (outer(g1) + (k * 0.5) * outer(g0 / s))
        return m, k


class DistDropout(Dropout):
    def forward(self, input: DistTensor) -> DistTensor:
        m, k = input
        # dropout
        if self.std != 0:
            assert self.target == "input"
            if k is None:
                k = torch.zeros_like(m).diag_embed()
            d = (m.square() + diag(k)) * self.std**2
            k = k + self._mean(d).diag_embed()
        # linear
        w, b = self.weight, self.bias
        m = F.linear(m, w, b)
        if k is not None:
            k = w @ k @ w.T
        return m, k


class DistMLP(MLP):
    _flatten = DistFlatten
    _activation = lambda _, config: get_dist_activation(config)
    _dropout = DistDropout


class DistCrossEntropyLoss(CrossEntropyLoss):
    def forward(self, outputs: DistTensor, targets: Tensor) -> Tensor:
        (m, k), i = outputs, targets
        L = cross_entropy(m, i)
        if k is not None:
            p = m.softmax(-1)
            h = p.diag_embed() + outer(p)
            L = L + (k * h).sum((-2, -1)) * 0.5
        return L
