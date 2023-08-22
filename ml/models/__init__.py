from __future__ import annotations

import torch
import torch.nn as nn

from ml.datasets import Dataset
from ml.models.mlp import *
from ml.models.dist_mlp import *
from ml.models.resnet import *
from ml.models.vgg import *
from ml.models.utils import *


def get_model(config: dict, dataset: Dataset) -> Model:
    name = config["name"]
    if name == "mlp":
        return MLP(config, dataset)
    if name == "reg-mlp":
        return RegMLP(config, dataset)
    if name == "dist-mlp":
        return DistMLP(config, dataset)
    raise NotImplementedError(name)


def get_optim(config: dict, model: nn.Module) -> torch.optim.Optimizer:
    name = config["name"]
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config["learning_rate"],
            momentum=config["momentum"],
        )
    raise NotImplementedError(name)


def get_loss_fn(config: dict) -> nn.Module:
    name = config["name"]
    if name == "ce":
        return CrossEntropyLoss(config)
    if name == "quad-ce":
        return QuadraticCrossEntropyLoss(config)
    if name == "mc-ce":
        return MonteCarloCrossEntropyLoss(config)
    raise NotImplementedError(name)
