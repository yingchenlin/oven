import torch
import torch.nn as nn
from torch import Tensor


from ml.datasets import ImageDataset
from ml.models.dropout import *
from ml.models.mlp import *
from ml.models.utils import *


class Regulator(Dropout):
    input: Tensor

    def forward(self, x: Tensor) -> Tensor:
        self.input = x
        return x


class RegMLP(MLP, Regulated):
    _dropout = Regulator

    def reg_loss(self, outputs: Tensor) -> Tensor:
        losses = torch.zeros(())
        if self.training:
            prob = outputs.softmax(-1)
            hess = prob.diag_embed() - outer(prob)
            jacob = torch.ones_like(prob).diag_embed()
            for m in reversed(self):
                if isinstance(m, nn.Linear):
                    jacob = jacob @ m.weight
                if isinstance(m, Regulator):
                    if m.std != 0.0:
                        var = means[m.mean](m.input.square()) * (m.std * m.std * 0.5)
                        losses = losses + torch.einsum(
                            "bij,bik,bjk,bk->b", hess, jacob, jacob, var
                        )
                    jacob = jacob * (m.input > 0).unsqueeze(-2)
        return losses
