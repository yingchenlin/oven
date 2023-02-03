import torch
from collections import namedtuple

if __package__ == "":
    from modules import Capture
else:
    from .modules import Capture

def get_metrics(config):
    return Metrics(config)

def get_ranks(x, i):
    return (x > x.gather(-1, i.unsqueeze(-1))).sum(-1)


class Metrics:

    def __init__(self, config):
        self.topks = config["topks"]
        self.reset()

    def reset(self):
        self.aggs = {}

    def add_losses(self, losses: torch.Tensor):
        self._agg("loss", "avg", losses)

    def add_ranks(self, outputs: torch.Tensor, targets: torch.Tensor):
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        ranks = get_ranks(outputs, targets)
        for k in self.topks:
            self._agg(f"top{k}", "avg", ranks < k)

    def add_states(self, model: torch.nn.Module, has_tensor: bool):
        for name, module in model.named_children():
            if isinstance(module, Capture):
                state = module.state
                grad = state.grad * state.shape[0] * state.shape[1]
                self._agg_norm(f"${name}.state", state)
                self._agg_norm(f"${name}.grad", grad)
                if has_tensor:
                    self._agg_moment(f"${name}.state", state)
                    self._agg_moment(f"${name}.grad", grad)

    def add_grads(self, outputs: torch.Tensor, targets: torch.Tensor, model: torch.nn.Module, has_tensor: bool):
        if has_tensor:
            x, i = outputs.detach(), targets
            p = x.softmax(-1)
            j = p - torch.nn.functional.one_hot(i, x.shape[-1])
            h = p.diag_embed() - p.unsqueeze(-2) * p.unsqueeze(-1)
            for name, m in reversed(list(model.named_children())[3:-1]):
                if isinstance(m, torch.nn.Linear):
                    w = m.weight.detach()
                    j = torch.einsum("sbi,ij->sbj", j, w)
                    h = torch.einsum("sbij,ik,jl->sbkl", h, w, w)
                if isinstance(m, Capture):
                    self._agg_moment(f"${name}.jacob", j)
                    self._agg(f"${name}.hess", "m1", h)
                    u = (m.state.detach() > 0)
                    j = j * u
                    h = h * u.unsqueeze(-2) * u.unsqueeze(-1)

    def add_params(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            self._agg_norm(f"${name}", param)

    def get(self):
        return {k: v.get() for k, v in self.aggs.items()}

    def _agg_norm(self, key, value):
        self._agg(key, "l0", value)
        self._agg(key, "l1", value)
        self._agg(key, "l2", value)

    def _agg_moment(self, key, value):
        self._agg(key, "m1", value)
        self._agg(key, "m2", value)

    def _agg(self, name, kind, values):
        if kind != "avg":
            name = f"{name}.{kind}"
        if name not in self.aggs:
            self.aggs[name] = Agg(kind)
        self.aggs[name].add(values)


class Agg:

    Transform = namedtuple("Transform", ["fwd", "inv", "is_scalar"])
    _transforms = {
        # scalar
        "avg": Transform(lambda x: x, lambda x: x, True),
        "l0": Transform(lambda x: x > 0, lambda x: x, True),
        "l1": Transform(lambda x: x.abs(), lambda x: x, True),
        "l2": Transform(lambda x: x.square(), lambda x: x.sqrt(), True),
        # tensor
        "m1": Transform(lambda x: x.sum((0, 1)), lambda x: x, False),
        "m2": Transform(lambda x: torch.einsum("sbi,sbj->ij", x, x), lambda x: x, False),
    }

    def __init__(self, kind):
        self.kind = kind
        self.num = 0
        self.sum = 0

    def add(self, x):
        t = self._transforms[self.kind]
        x = x.detach()
        if t.is_scalar:
            x = x.flatten()
            self.num += len(x)
            self.sum += t.fwd(x).sum()
        else:
            self.num += x.shape[0] * x.shape[1]
            self.sum += t.fwd(x)

    def get(self):
        t = self._transforms[self.kind]
        x = t.inv(self.sum / self.num)
        if t.is_scalar:
            return x.item()
        else:
            return x.cpu()
