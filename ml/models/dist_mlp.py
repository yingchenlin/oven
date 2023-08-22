from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ml.models.mlp import *
from ml.models.utils import *


def get_dist_activation(config: dict) -> nn.Module:
    name = config["name"]
    if name == "relu":
        return DistReLU()
    raise NotImplementedError(name)


class DistFlatten(nn.Flatten):
    def forward(self, input: Tensor) -> DistTensor:
        return super().forward(input), None


class DistReLU(nn.Module):
    def forward(self, input: DistTensor) -> DistTensor:
        m, k = input
        if k is None:
            return F.relu(m), None
        s = diag(k).sqrt() + 1e-8
        z = m / s
        g0 = gauss_pdf(z)
        g1 = gauss_cdf(z)
        m = m * g1 + s * g0
        k = k * (outer(g1) + (k * 0.5) * outer(g0 / s))
        return m, k


class DistDropout(Dropout):
    def forward(self, input: DistTensor) -> DistTensor:
        m, k = input
        if self.training and self.std != 0.0:
            if k is None:
                d = means[self.mean](m.square()) * self.std**2
                k = d.diag_embed()
            else:
                d = means[self.mean](m.square() + diag(k)) * self.std**2
                k = k + d.diag_embed()
        return m, k


class DistLinear(nn.Linear):
    def forward(self, input: DistTensor) -> DistTensor:
        m, k = input
        w, b = self.weight, self.bias
        m = F.linear(m, w, b)
        if k is not None:
            k = w @ k @ w.T
        return m, k


class DistMLP(MLP):
    _flatten = DistFlatten
    _dropout = DistDropout
    _linear = DistLinear
    _activation = lambda _, config: get_dist_activation(config)


class QuadraticCrossEntropyLoss(CrossEntropyLoss):
    def forward(self, outputs: DistTensor, targets: Tensor) -> Tensor:
        (m, k), i = outputs, targets
        losses = cross_entropy(m, i)
        if k is not None:
            prob = m.softmax(-1)
            hess = prob.diag_embed() - outer(prob)
            losses = losses + (k * hess).sum((-2, -1)) * 0.5
        return losses


class MonteCarloCrossEntropyLoss(CrossEntropyLoss):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.num_samples = config["num_samples"]

    def extra_repr(self) -> str:
        return f"num_samples={self.num_samples}"

    def forward(self, outputs: DistTensor, targets: Tensor) -> Tensor:
        (m, k), i = outputs, targets
        if k is None:
            return cross_entropy(m, i)
        x = sample(m, k, self.num_samples)
        x = x.swapaxes(-2, -1)
        i = i.unsqueeze(-1)
        return cross_entropy(x, i).mean(-1)
