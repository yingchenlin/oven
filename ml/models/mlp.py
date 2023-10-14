import numpy as np
import torch.nn as nn

from ml.datasets import ImageDataset
from ml.models.dropout import Dropout, DropoutConfig
from ml.models.utils import Capture


def get_activation(config: dict) -> nn.Module:
    name = config["name"]
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise NotImplementedError(name)


class MLP(nn.Sequential):
    _flatten = nn.Flatten
    _dropout = Dropout
    _linear = nn.Linear
    _activation = lambda _, config: get_activation(config)

    def __init__(self, config: dict, dataset: ImageDataset) -> None:
        num_layers: int = config["num_layers"]
        hidden_dim: int = config["hidden_dim"]
        output_dim: int = int(np.prod(dataset.input_shape))
        dropout_config = DropoutConfig(config["dropout"], config["dropout"]["std"])

        layers = []
        layers.append(self._flatten(start_dim=-len(dataset.input_shape)))
        for i in range(num_layers):
            input_dim, output_dim = output_dim, hidden_dim
            layers.append(self._dropout(dropout_config[i]))
            layers.append(self._linear(input_dim, output_dim))
            layers.append(Capture())
            layers.append(self._activation(config["activation"]))
            layers.append(Capture())
        input_dim, output_dim = output_dim, dataset.num_classes
        layers.append(self._dropout(dropout_config[num_layers]))
        layers.append(self._linear(input_dim, output_dim))
        layers.append(Capture())

        super().__init__(*layers)
