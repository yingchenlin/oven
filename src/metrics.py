import torch
from collections import namedtuple

if __package__ == "":
    from modules import Capture
else:
    from .modules import Capture

def get_metrics(config):
    return Metrics(config)

def get_ranks(x, i):
    x_i = x.gather(-1, i.unsqueeze(-1))
    return (~(x < x_i)).sum(-1) # handles nan


class Metrics:

    def __init__(self, config):
        self.topks = config["topks"]
        self.reset()

    def reset(self):
        self.aggs = {}

    def add_losses(self, losses: torch.Tensor):
        self._agg_scalar("loss", "avg", losses)

    def add_ranks(self, outputs: torch.Tensor, targets: torch.Tensor):
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        ranks = get_ranks(outputs, targets)
        for k in self.topks:
            self._agg_scalar(f"top{k}", "avg", ranks <= k)

    def add_states(self, model: torch.nn.Module):
        for name, module in model.named_children():
            if isinstance(module, Capture):
                state = module.state
                grad = state.grad * state.shape[0] * state.shape[1]
                self._agg_norms(f"${name}.state", state)
                self._agg_norms(f"${name}.grad", grad)

    def add_params(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            self._agg_norms(f"${name}", param)

    def get(self):
        return {k: v.get() for k, v in self.aggs.items()}

    def _agg_norms(self, key, value):
        self._agg_scalar(key, "l0", value)
        self._agg_scalar(key, "l1", value)
        self._agg_scalar(key, "l2", value)

    def _agg_moments(self, key, value):
        self._agg_tensor(key, "m1", value)
        self._agg_tensor(key, "m2", value)

    def _agg_scalar(self, name, kind, values):
        if kind != "avg":
            name = f"{name}.{kind}"
        if name not in self.aggs:
            self.aggs[name] = ScalarAgg(kind)
        self.aggs[name].add(values)

    def _agg_tensor(self, name, kind, values):
        if kind != "avg":
            name = f"{name}.{kind}"
        if name not in self.aggs:
            self.aggs[name] = TensorAgg(kind)
        self.aggs[name].add(values)


Transform = namedtuple("Transform", ["fwd", "inv"])

class ScalarAgg:

    _transforms = {
        "avg": Transform(lambda x: x, lambda x: x),
        "l0": Transform(lambda x: x > 0, lambda x: x),
        "l1": Transform(lambda x: x.abs(), lambda x: x),
        "l2": Transform(lambda x: x.square(), lambda x: x.sqrt()),
    }

    def __init__(self, kind):
        self.kind = kind
        self.num = 0
        self.sum = 0

    def add(self, values):
        transform = self._transforms[self.kind]
        values = values.detach().flatten()
        self.num += len(values)
        self.sum += transform.fwd(values).sum()

    def get(self):
        transform = self._transforms[self.kind]
        mean = self.sum / (self.num + 1e-8)
        return transform.inv(mean).item()


class TensorAgg:

    _transforms = {
        "m1": Transform(lambda x: x.sum((0, 1)), lambda x: x),
        "m2": Transform(lambda x: torch.einsum("sbi,sbj->ij", x, x), lambda x: x),
    }

    def __init__(self, kind):
        self.kind = kind
        self.num = 0
        self.sum = 0

    def add(self, values):
        transform = self._transforms[self.kind]
        values = values.detach()
        self.num += values.shape[0] * values.shape[1]
        self.sum += transform.fwd(values)

    def get(self):
        transform = self._transforms[self.kind]
        mean = self.sum / self.num
        return transform.inv(mean).cpu()
