from torch import nn, Tensor

from ml.models.utils import *


class CrossEntropyLoss(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()

    def forward(self, outputs: Tensor, targets: Tensor):
        return cross_entropy(outputs, targets)


class HingeLoss(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()

    def forward(self, outputs: Tensor, targets: Tensor):
        return nn.functional.multi_margin_loss(outputs, targets, reduction="none")
