from __future__ import annotations

import torch
from dataclasses import dataclass
from torch import nn, Tensor
from typing import Callable, Dict, Mapping, Union

from ml.models.utils import Capture, sample


def get_metrics(config: dict) -> Metrics:
    return Metrics(config)


def get_ranks(x: Tensor, i: Tensor) -> Tensor:
    x_i = x.gather(-1, i.unsqueeze(-1))
    return (~(x < x_i)).sum(-1)  # handles nan


class Metrics:
    def __init__(self, config):
        self.topks = config["topks"]
        self.reset()

    def reset(self):
        self.scalars: Dict[str, ScalarAgg] = {}
        self.tensors: Dict[str, TensorAgg] = {}

    def add_losses(self, losses: Tensor):
        self._agg_scalar("loss", "avg", losses)

    def add_ranks(self, outputs: Tensor, targets: Tensor):
        if isinstance(outputs, tuple):
            m, k = outputs
            outputs = sample(m, k, 1).squeeze(-1)
        ranks = get_ranks(outputs, targets)
        for k in self.topks:
            self._agg_scalar(f"top{k}", "avg", ranks <= k)

    def add_states(self, model: nn.Module, moments=False) -> None:
        for name, module in model.named_children():
            if isinstance(module, Capture):
                state, grad = module.get()
                self._agg_norms(f"${name}.state", state)
                self._agg_norms(f"${name}.grad", grad)
                if moments:
                    self._agg_moments(f"${name}.sign", state > 0)
                    self._agg_moments(f"${name}.state", state)
                    self._agg_moments(f"${name}.grad", grad)

    def add_params(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            self._agg_norms(f"${name}", param)

    def get_scalars(self) -> Mapping[str, float]:
        return {k: v.get() for k, v in self.scalars.items()}

    def get_tensors(self) -> Mapping[str, Tensor]:
        return {k: v.get() for k, v in self.tensors.items()}

    def _agg_norms(self, key: str, value: Tensor) -> None:
        self._agg_scalar(key, "l0", value)
        self._agg_scalar(key, "l1", value)
        self._agg_scalar(key, "l2", value)

    def _agg_moments(self, key: str, value: Tensor) -> None:
        self._agg_tensor(key, "m1", value)
        self._agg_tensor(key, "m2", value)

    def _agg_scalar(self, key, kind, values):
        name = f"{key}.{kind}"
        if kind == "avg":
            name = key
        if name not in self.scalars:
            self.scalars[name] = ScalarAgg(kind)
        self.scalars[name].add(values)

    def _agg_tensor(self, key, kind, values):
        name = f"{key}.{kind}"
        if name not in self.tensors:
            self.tensors[name] = TensorAgg(kind)
        self.tensors[name].add(values)


@dataclass
class Transform:
    fwd: Callable[[Tensor], Tensor]
    inv: Callable[[Tensor], Tensor]


def noop(x: Tensor) -> Tensor:
    return x


def sign(x: Tensor) -> Tensor:
    return x > 0


def outer(x: Tensor) -> Tensor:
    return x.unsqueeze(-2) * x.unsqueeze(-1)


class Agg:
    _transforms: Mapping[str, Transform]

    def __init__(self, name_or_transform: Union[str, Transform]) -> None:
        if isinstance(name_or_transform, str):
            name_or_transform = self._transforms[name_or_transform]
        self.transform: Transform = name_or_transform
        self.num: int = 0
        self.sum: Tensor = torch.zeros(())


class ScalarAgg(Agg):
    _transforms = {
        "avg": Transform(noop, noop),
        "l0": Transform(sign, noop),
        "l1": Transform(torch.abs, noop),
        "l2": Transform(torch.square, torch.sqrt),
    }

    def add(self, values: Tensor) -> None:
        values = values.detach()
        self.num += values.numel()
        self.sum = self.sum + self.transform.fwd(values).sum()

    def get(self) -> float:
        mean = self.sum / (self.num + 1e-8)
        return self.transform.inv(mean).item()


class TensorAgg(Agg):
    _transforms = {
        "m1": Transform(noop, noop),
        "m2": Transform(outer, noop),
    }

    def add(self, values: Tensor) -> None:
        values = values.detach()
        self.num += values.shape[0] * values.shape[1]
        self.sum = self.sum + self.transform.fwd(values).sum((0, 1))

    def get(self) -> Tensor:
        mean = self.sum / (self.num + 1e-8)
        return self.transform.inv(mean).cpu()
