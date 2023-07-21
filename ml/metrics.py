from __future__ import annotations

import torch
from torch import nn, Tensor
from collections import namedtuple
from typing import Dict

from ml.modules import Capture, sample


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
        self.aggs: Dict[str, ScalarAgg] = {}

    def add_losses(self, losses: Tensor):
        self._agg_scalar("loss", "avg", losses)

    def add_ranks(self, outputs: Tensor, targets: Tensor):
        if isinstance(outputs, tuple):
            m, k = outputs
            outputs = sample(m, k)
        ranks = get_ranks(outputs, targets)
        for k in self.topks:
            self._agg_scalar(f"top{k}", "avg", ranks <= k)

    def add_states(self, model: nn.Module) -> None:
        for name, module in model.named_children():
            if isinstance(module, Capture):
                if module.state is not None:
                    state = module.state
                    self._agg_norms(f"${name}.state", state)
                    if state.grad is not None:
                        grad = state.grad.nan_to_num() * state.shape[0] * state.shape[1]
                        self._agg_norms(f"${name}.grad", grad)

    def add_params(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            self._agg_norms(f"${name}", param)

    def get(self) -> Dict[str, float]:
        return {k: v.get() for k, v in self.aggs.items()}

    def _agg_norms(self, key: str, value: Tensor) -> None:
        self._agg_scalar(key, "l0", value)
        self._agg_scalar(key, "l1", value)
        self._agg_scalar(key, "l2", value)

    """
    def _agg_moments(self, key: str, value: Tensor) -> None:
        self._agg_tensor(key, "m1", value)
        self._agg_tensor(key, "m2", value)
    """

    def _agg_scalar(self, name, kind, values):
        if kind != "avg":
            name = f"{name}.{kind}"
        if name not in self.aggs:
            self.aggs[name] = ScalarAgg(kind)
        self.aggs[name].add(values)

    """
    ef _agg_tensor(self, name, kind, values):
        if kind != "avg":
            name = f"{name}.{kind}"
        if name not in self.aggs:
            self.aggs[name] = TensorAgg(kind)
        self.aggs[name].add(values)
    """


Transform = namedtuple("Transform", ["fwd", "inv"])


class Agg:
    _transforms: Dict[str, Transform]

    def __init__(self, kind: str) -> None:
        assert kind in self._transforms
        self.kind = kind
        self.num = 0
        self.sum = 0


class ScalarAgg(Agg):
    _transforms = {
        "avg": Transform(lambda x: x, lambda x: x),
        "l0": Transform(lambda x: x > 0, lambda x: x),
        "l1": Transform(lambda x: x.abs(), lambda x: x),
        "l2": Transform(lambda x: x.square(), lambda x: x.sqrt()),
    }

    def add(self, values: Tensor) -> None:
        transform = self._transforms[self.kind]
        values = values.detach().flatten()
        self.num += len(values)
        self.sum += transform.fwd(values).sum()

    def get(self) -> float:
        transform = self._transforms[self.kind]
        mean = self.sum / (self.num + 1e-8)
        return transform.inv(mean).item()


class TensorAgg(Agg):
    _transforms = {
        "m1": Transform(lambda x: x.sum((0, 1)), lambda x: x),
        "m2": Transform(lambda x: torch.einsum("sbi,sbj->ij", x, x), lambda x: x),
    }

    def add(self, values: Tensor) -> None:
        transform = self._transforms[self.kind]
        values = values.detach()
        self.num += values.shape[0] * values.shape[1]
        self.sum += transform.fwd(values)

    def get(self) -> Tensor:
        transform = self._transforms[self.kind]
        mean = self.sum / self.num
        return transform.inv(mean).cpu()
