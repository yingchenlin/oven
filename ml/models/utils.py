import abc
import numpy as np
import torch
from torch import Tensor, nn
from typing import Optional, Tuple, Union


def outer(x: Tensor) -> Tensor:
    return x.unsqueeze(-1) * x.unsqueeze(-2)


def diag(x: Tensor) -> Tensor:
    return x.diagonal(0, -2, -1)


def gauss_pdf(z: Tensor) -> Tuple:
    return (z.square() * -0.5).exp() * np.sqrt(1 / (np.pi * 2))


def gauss_cdf(z: Tensor) -> Tensor:
    return ((z * np.sqrt(0.5)).erf() + 1) * 0.5


def cross_entropy(x: Tensor, i: Tensor) -> Tensor:
    return x.logsumexp(-1) - x.gather(-1, i.unsqueeze(-1)).squeeze(-1)


class CrossEntropyLoss(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()

    def forward(self, outputs, targets):
        return cross_entropy(outputs, targets)


DistTensor = Tuple[Tensor, Optional[Tensor]]


def sample(m: Tensor, k: Optional[Tensor], n: int) -> Tensor:
    if k is None:
        return m
    q, _ = torch.linalg.cholesky_ex(k)
    d = torch.randn(m.shape + (n,), device=m.device)
    return m.unsqueeze(-1) + q @ d


class Capture(nn.Module):
    state: Tensor

    def forward(self, input: Union[Tensor, DistTensor]) -> Union[Tensor, DistTensor]:
        if isinstance(input, Tensor):
            self.state = input
        else:
            m, k = input
            self.state = m
        self.state.retain_grad()
        return input

    def get(self) -> Tuple[Tensor, Tensor]:
        assert self.state.grad is not None
        grad = self.state.grad.nan_to_num() * np.prod(self.state.shape[:-1])
        return self.state, grad


class Model(abc.ABC):
    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def reg_loss(self, outputs: Tensor) -> Tensor:
        return torch.zeros(())
