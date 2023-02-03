import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def get_model(config, dataset):
    name = config["name"]
    if name == "mlp":
        return MLP(config, dataset)
    if name == "dist-mlp":
        return DistMLP(config, dataset)
    if name == "resnet":
        return torchvision.models.resnet18(num_classes=dataset.num_classes)
    raise NotImplementedError()

def get_optim(config, model):
    name = config["name"]
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config["learning_rate"],
            momentum=config["momentum"],
        )
    raise NotImplementedError()

def get_loss_fn(config):
    name = config["name"]
    if name == "ce":
        return CrossEntropyLoss(config)
    if name == "n-ce":
        return NormCrossEntropyLoss(config)
    if name == "dn-ce":
        return DropoutNormCrossEntropyLoss(config)
    if name == "dist-ce":
        return DistCrossEntropyLoss(config)
    raise NotImplementedError()

def get_activation(config):
    name = config["name"]
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise NotImplementedError()


class Capture(nn.Module):

    def forward(self, input):
        self.state = input
        if isinstance(self.state, tuple):
            self.state = self.state[0]
        self.state.retain_grad()
        return input


class Loss(nn.Module):

    def __init__(self, config):
        super().__init__()

    def forward(self, inputs, outputs, targets, model: nn.Module):
        raise NotImplementedError()


class RegLoss(Loss):

    def __init__(self, config):
        super().__init__(config)
        self.weight = config["weight"]

# MLP


class Dropout(nn.Module):

    _sampe_like = {
        "bernoulli": lambda x, std: 
            torch.bernoulli(torch.full_like(x, 1 / (1 + std * std))),
        "uniform": lambda x, std: 
            (torch.rand_like(x) - 0.5) * (std * 2 * np.sqrt(3)) + 1,
        "normal": lambda x, std:
            torch.randn_like(x) * std + 1,
    }

    def __init__(self, config):
        super().__init__()
        self.std = config["std"]
        self.dist = config["dist"]

    def extra_repr(self):
        return f"std={self.std} dist={self.dist}"

    def forward(self, x):
        if self.training:
            x = x * self._sampe_like[self.dist](x, self.std)
        return x


class MLP(nn.Sequential):

    _flatten = nn.Flatten
    _linear = nn.Linear
    _activation = lambda _, config: get_activation(config)
    _dropout = Dropout

    def __init__(self, config, dataset):

        layers = []
        layers.append(self._flatten(start_dim=-3))
        output_dim = np.prod(dataset.input_shape)
        for i in range(config["num_layers"]):
            input_dim, output_dim = output_dim, config["hidden_dim"]
            layers.append(self._linear(input_dim, output_dim))
            layers.append(self._activation(config["activation"]))
            layers.append(Capture())
            layers.append(self._dropout(config["dropout"]))
        input_dim, output_dim = output_dim, dataset.num_classes
        layers.append(self._linear(input_dim, output_dim))
        layers.append(Capture())

        super().__init__(*layers)


class CrossEntropyLoss(Loss):

    def forward(self, inputs, outputs, targets, model: nn.Module):
        return cross_entropy(outputs, targets)


class NormCrossEntropyLoss(RegLoss):

    def forward(self, inputs, outputs, targets, model: nn.Module):
        return cross_entropy(outputs, targets) + outputs.square().mean(-1) * self.weight


class DropoutNormCrossEntropyLoss(RegLoss):

    def forward(self, inputs, outputs, targets, model: nn.Module):
        mean = outputs
        if model.training:
            model.eval()
            mean = model(inputs)
            model.train()
        diff = outputs - mean
        return cross_entropy(mean, targets) + diff.square().mean(-1) * self.weight


# DistMLP

def outer(x):
    return x.unsqueeze(-1) * x.unsqueeze(-2)

def diag(x):
    return x.diagonal(0, -2, -1)

def gaussian(z):
    g0 = (z.square() * -0.5).exp() * np.sqrt(1 / (np.pi * 2))
    g1 = ((z * np.sqrt(0.5)).erf() + 1) * 0.5
    return g0, g1

def cross_entropy(x, i, dim=-1):
    return x.logsumexp(dim) - x.gather(dim, i.unsqueeze(dim)).squeeze(dim)

def get_dist_activation(config):
    name = config["name"]
    if name == "relu":
        return DistReLU()
    raise NotImplementedError()


class DistFlatten(nn.Flatten):

    def forward(self, input):
        return super().forward(input), None


class DistLinear(nn.Linear):

    def forward(self, input):
        m, k = input
        mp = super().forward(m)
        kp = self._cov(k)
        return mp, kp

    def _cov(self, k):
        if k == None:
            return None
        w = self.weight
        return w @ k @ w.T


class DistReLU(nn.Module):

    def forward(self, input):
        m, k = input
        if k == None:
            return F.relu(m), None
        s = diag(k).sqrt() + 1e-8
        g0, g1 = gaussian(m / s)
        mp = m * g1 + s * g0
        kp = k * (outer(g1) + (k * 0.5) * outer(g0 / s))
        return mp, kp


class DistDropout(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.std = config["std"]

    def extra_repr(self):
        return f"std={self.std}"

    def forward(self, input):
        m, k = input
        mp, kp = m, k
        if self.std != 0:
            d = m.square()
            if k is not None:
                d = d + diag(k)
            v = self.std * self.std
            kp = (d * v).diag_embed()
            if k is not None:
                kp = kp + k
        return mp, kp


class DistMLP(MLP):

    _flatten = DistFlatten
    _linear = DistLinear
    _activation = lambda _, config: get_dist_activation(config)
    _dropout = DistDropout


class DistCrossEntropyLoss(Loss):

    def forward(self, inputs, outputs, targets):
        (m, k), i = outputs, targets
        L = cross_entropy(m, i)
        if k is not None:
            p = m.softmax(-1)
            h = p.diag_embed() - outer(p)
            L = L + (k * h).sum((-2, -1)) * 0.5
        return L
