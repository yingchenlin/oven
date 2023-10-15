import json
import os
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import faiss
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import torch
from torch import Tensor

from ml.engine import Engine
from ml.models import Capture, Dropout, entropy, outer, one_hot
from ml.metrics import TensorAgg


def load_data(path, gen):
    if not os.path.exists(path):
        data = gen()
        torch.save(data, path)
    data = torch.load(path)
    return data


def load_engine(
    path: str,
    epoch: int,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = True,
) -> Engine:
    with open(f"{path}/config.json") as file:
        config = json.load(file)
    config["dataset"] = {
        "name": config["dataset"]["name"],
        "transform": {"normalize": {"mean": 0.5, "std": 0.2}},
        "train": {"batch_size": 100, "shuffle": False},
        "test": {"batch_size": 100, "shuffle": False},
    }
    engine = Engine(config, seed, device, verbose)
    engine.load(f"{path}/checkpoint-{epoch}.pt")
    return engine


def get_modules(engine: Engine) -> Mapping[int, Capture]:
    return {
        layer: module
        for layer, module in enumerate(engine.model.children())
        if isinstance(module, Capture)
    }


@dataclass
class Samples:
    states: Mapping[int, Tensor]
    outputs: Tensor
    predicts: Tensor
    targets: Tensor
    losses: Tensor


def get_samples(
    engine: Engine,
    training_data: bool,
    training_mode: bool,
) -> Samples:
    engine.model.train(mode=training_mode)
    engine.loss_fn.train(mode=training_mode)

    modules = get_modules(engine)
    states_list_map = {layer: [] for layer in modules}
    outputs_list = []
    predicts_list = []
    targets_list = []
    losses_list = []
    for _, outputs, targets, losses in engine.loop(train=training_data):
        predicts = outputs.argmax(-1)
        for layer, module in modules.items():
            states_list_map[layer].append(module.state.detach().cpu())
        outputs_list.append(outputs.detach().cpu())
        predicts_list.append(predicts.detach().cpu())
        targets_list.append(targets.detach().cpu())
        losses_list.append(losses.detach().cpu())

    return Samples(
        states={
            layer: torch.concat(states_list)
            for layer, states_list in states_list_map.items()
        },
        outputs=torch.concat(outputs_list),
        predicts=torch.concat(predicts_list),
        targets=torch.concat(targets_list),
        losses=torch.concat(losses_list),
    )


@dataclass
class Neighbours:
    distances: Tensor
    indices: Tensor


def get_neighbours(
    key_samples: Samples, query_samples: Samples, layer: int, topk: int
) -> Neighbours:
    keys = key_samples.states[layer].numpy()
    queries = query_samples.states[layer].numpy()
    index = faiss.IndexFlat(keys.shape[-1])
    index.add(keys)
    distances, indices = index.search(queries, k=topk)
    return Neighbours(
        distances=torch.tensor(distances),
        indices=torch.tensor(indices),
    )


@dataclass
class Moments:
    x1: Tensor
    x2: Tensor
    g1: Tensor
    g2: Tensor
    k: Tensor


def get_moments(
    engine: Engine,
    training_data: bool,
    training_mode: bool = False,
) -> Mapping[int, Moments]:
    modules = list(engine.model.children())
    layers = [i for i, m in enumerate(modules) if isinstance(m, Capture)]
    x1_aggs = {layer: TensorAgg("m1") for layer in layers}
    x2_aggs = {layer: TensorAgg("m2") for layer in layers}
    g1_aggs = {layer: TensorAgg("m1") for layer in layers}
    g2_aggs = {layer: TensorAgg("m2") for layer in layers}
    k_aggs = {layer: TensorAgg("m1") for layer in layers}

    engine.model.train(mode=training_mode)
    engine.loss_fn.train(mode=training_mode)
    for _, outputs, targets, _ in engine.loop(train=training_data):
        x = outputs
        p = x.softmax(-1)
        v = p - one_hot(targets, p)
        h = (p.diag_embed() - outer(p)) * 0.5
        j = torch.ones_like(p).diag_embed()
        last_module = None
        for layer, module in reversed(list(enumerate(modules))):
            if isinstance(module, torch.nn.Linear):
                j = j @ module.weight
            elif isinstance(module, torch.nn.ReLU):
                assert isinstance(last_module, Capture)
                j = j * (x > 0).unsqueeze(-2)
            elif isinstance(module, Capture):
                x = module.state
                g = torch.einsum("bi,bij->bj", v, j)
                k = torch.einsum("bij,bik,bjl->bkl", h, j, j)
                x1_aggs[layer].add(x)
                x2_aggs[layer].add(x)
                g1_aggs[layer].add(g)
                g2_aggs[layer].add(g)
                k_aggs[layer].add(k)
                if layer == min(layers):
                    break
            last_module = module

    return {
        layer: Moments(
            x1=x1_aggs[layer].get(),
            x2=x2_aggs[layer].get(),
            g1=g1_aggs[layer].get(),
            g2=g2_aggs[layer].get(),
            k=k_aggs[layer].get(),
        )
        for layer in layers
    }


@dataclass
class Dist:
    m: Tensor
    k: Tensor
    u: Tensor
    s: Tensor
    v: Tensor


def get_dists(samples: Samples) -> Mapping[int, Dist]:
    dists = {}
    for layer, x in samples.states.items():
        m = x.sum(0) / len(x)
        k = (x.T @ x / len(x)) - m[:, None] * m[None, :]
        u, s, v = torch.linalg.svd(k)
        dists[layer] = Dist(m, k, u, s, v)
    return dists


@dataclass
class Coord:
    origin: Tensor
    x_axis: Tensor
    y_axis: Tensor
    radius: float

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.origin + self.radius * (
            x.unsqueeze(-1) * self.x_axis + y.unsqueeze(-1) * self.y_axis
        )

    def inverse(self, z: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        v = (z - self.origin) / self.radius
        x = v @ self.x_axis
        y = v @ self.y_axis
        d = v - (x.unsqueeze(-1) * self.x_axis + y.unsqueeze(-1) * self.y_axis)
        z = d.square().sum(-1).sqrt()
        return x, y, z


def basis(*inputs: Tensor) -> Tuple[Tensor, ...]:
    outputs = list(inputs)
    for i, u in enumerate(outputs):
        for v in outputs[:i]:
            u = u - v * (u.dot(v) / v.dot(v))  # orthogonal
        u = u / u.dot(u).sqrt()  # normal
        outputs[i] = u
    return tuple(outputs)


def get_coord(
    dist: Dist,
    kind: str,
    scale: float,
) -> Coord:
    m = dist.m
    u = dist.u
    r = dist.s.sum().sqrt().item() * scale

    if kind == "basis":
        v0, v1 = u[:, 0], u[:, 1]
    elif kind == "random":
        v0, v1 = basis(u[:, 0], torch.randn_like(u[:, 0]))
    elif kind == "origin":
        v0, v1 = basis(u[:, 0], m)
    elif kind == "ortho":
        _, v0, v1 = basis(m, u[:, 0], u[:, 1])
    else:
        raise ValueError(f"unrecognized kind [{kind}]")

    return Coord(m, v0, v1, r)


def plot_coord(coord: Coord, ax: plt.Axes):
    p = torch.stack([coord.origin, torch.zeros_like(coord.origin)])
    x, y, z = coord.inverse(p)
    ax.scatter(x, y, marker="o", c="w")
    ax.scatter(x, y, marker="x", c="k")
    ax.grid()


def plot_slice1d(
    engine: Engine,
    layer: int,
    coord: Coord,
    axis: str,
    label: str,
    size: int,
    ax: Optional[plt.Axes] = None,
):
    assert isinstance(engine.model, torch.nn.Sequential)
    modules = list(engine.model.children())[layer:]
    modules = [m for m in modules if not isinstance(m, Capture)]
    model = torch.nn.Sequential(*modules)

    engine.model.train(False)

    t = (torch.arange(size) + 0.5) / size * 2 - 1
    if axis == "x":
        inputs = coord.forward(t[None, :], torch.zeros((1, 1)))
    elif axis == "y":
        inputs = coord.forward(torch.zeros((1, 1)), t[None, :])
    else:
        raise NotImplementedError
    with torch.no_grad():
        outputs = model(inputs.to(engine.device)).detach().cpu()
    s = entropy(outputs.squeeze(0)) / torch.tensor(outputs.shape[-1]).log()

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(t, s, label=label)


def plot_samples1d(
    samples: Samples,
    layer: int,
    coord: Coord,
    axis: str,
    label: str,
    distance: float = torch.inf,
    ratio: float = 0,
    ax: Optional[plt.Axes] = None,
):
    x, y, z = coord.inverse(samples.states[layer])
    if axis == "x":
        t = x
    elif axis == "y":
        t = y
    else:
        raise NotImplementedError

    i = torch.arange(len(z))
    if ratio > 0:
        i = z.topk(int(len(z) * ratio), largest=False, sorted=False).indices
    if distance < torch.inf:
        i = i[z[i] < distance]

    if ax is None:
        _, ax = plt.subplots()
    ax.hist(t[i], density=True, bins=50, range=(-1, 1), histtype="step", label=label)
    ax.set_xlim(-1, 1)


def plot_slice(
    engine: Engine,
    layer: int,
    coord: Coord,
    size: int,
    color: str,
    palette: str = "cubehelix",
    ax: Optional[plt.Axes] = None,
):
    assert isinstance(engine.model, torch.nn.Sequential)
    modules = list(engine.model.children())[layer:]
    modules = [m for m in modules if not isinstance(m, Capture)]
    model = torch.nn.Sequential(*modules)

    engine.model.train(False)

    s = torch.arange(size + 1) / size * 2 - 1
    t = (torch.arange(size) + 0.5) / size * 2 - 1
    inputs = coord.forward(t[None, :], t[:, None])
    with torch.no_grad():
        outputs = model(inputs.to(engine.device)).detach().cpu()

    n = outputs.shape[-1]
    if color == "label":
        k = torch.argmax(outputs, dim=-1) / n
    elif color == "entropy":
        k = entropy(outputs) / torch.tensor(n).log()

    if ax is None:
        _, ax = plt.subplots()
    cmap = sns.color_palette(palette, as_cmap=True)
    ax.pcolormesh(s, s, cmap(k))


def plot_samples(
    samples: Samples,
    layer: int,
    coord: Coord,
    distance: float = torch.inf,
    ratio: float = 0,
    palette: str = "cubehelix",
    color: str = "target",
    ax: Optional[plt.Axes] = None,
):
    x, y, z = coord.inverse(samples.states[layer])
    n = samples.outputs.shape[-1]
    if color == "target":
        k = samples.targets / n
    elif color == "loss":
        k = samples.losses / torch.tensor(n).log()
    elif color == "density":
        k = torch.zeros_like(z)
    else:
        raise ValueError(f"illegal color [{color}]")

    i = torch.arange(len(z))
    if ratio > 0:
        i = z.topk(int(len(z) * ratio), largest=False, sorted=False).indices
    if distance < torch.inf:
        i = i[z[i] < distance]

    if ax is None:
        _, ax = plt.subplots()

    cmap = sns.color_palette(palette, as_cmap=True)
    if color == "density":
        ax.hist2d(x[i], y[i], cmap=cmap, bins=(50, 50), range=((-1, 1), (-1, 1)))
    else:
        ax.scatter(x[i], y[i], c=cmap(k[i]), s=10, edgecolors="grey")

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
