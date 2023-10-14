import torch.nn as nn

from ml.datasets.image import ImageDataset
from ml.models.dropout import Dropout, DropoutConfig


"""
resnet18:
    block: basic_block
    num_blocks: [2, 2, 2, 2]
resnet34:
    block: basic_block
    num_blocks: [3, 4, 6, 3]
resnet50:
    block: bottleneck
    num_blocks: [3, 4, 6, 3]
resnet101:
    block: bottleneck
    num_blocks: [3, 4, 23, 3]
resnet152:
    block: bottleneck
    num_blocks: [3, 8, 36, 3]
"""


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, dropout_config: DropoutConfig):
        super().__init__()
        out_planes = planes * self.expansion
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(),
            Dropout(dropout_config[0]),
            nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            Dropout(dropout_config[1]),
            nn.Conv2d(planes, planes, 3, padding=1, bias=False),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                self.block[0],  # nn.BatchNorm2d(in_planes),
                nn.ReLU(),
                Dropout(dropout_config[2]),
                nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False),
            )

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride, dropout_config: DropoutConfig):
        super().__init__()
        out_planes = planes * self.expansion
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(),
            Dropout(dropout_config[0]),
            nn.Conv2d(in_planes, planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            Dropout(dropout_config[1]),
            nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            Dropout(dropout_config[2]),
            nn.Conv2d(planes, out_planes, 1, bias=False),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                self.block[0],  # nn.BatchNorm2d(in_planes),
                nn.ReLU(),
                Dropout(dropout_config[3]),
                nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False),
            )

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


blocks = {
    "basic_block": BasicBlock,
    "bottleneck": Bottleneck,
}


class ResNet(nn.Sequential):
    def __init__(self, config: dict, dataset: ImageDataset):
        block = blocks[config["block"]]
        num_blocks = config["num_blocks"]
        dropout_config = DropoutConfig(config["dropout"], config["dropout"]["std"])
        n = block.expansion
        super().__init__(
            Dropout(dropout_config[0]),
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            self._make_layer(block, 64, 64, num_blocks[0], 1, dropout_config[1]),
            self._make_layer(block, 64 * n, 128, num_blocks[1], 2, dropout_config[2]),
            self._make_layer(block, 128 * n, 256, num_blocks[2], 2, dropout_config[3]),
            self._make_layer(block, 256 * n, 512, num_blocks[3], 2, dropout_config[4]),
            nn.AvgPool2d(4),
            nn.Flatten(),
            Dropout(dropout_config[5]),
            nn.Linear(512 * n, dataset.num_classes),
        )

    def _make_layer(
        self,
        block,
        in_planes,
        planes,
        num_blocks,
        stride,
        dropout_config: DropoutConfig,
    ):
        n = block.expansion
        return nn.Sequential(
            block(in_planes, planes, stride, dropout_config[0]),
            *[
                block(planes * n, planes, 1, dropout_config[i])
                for i in range(1, num_blocks)
            ],
        )
