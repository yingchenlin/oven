import torch
import torch.nn as nn

from ml.datasets.image import ImageDataset

# fmt: off
cfg = {
    "VGG11": [64,     "M", 128,      "M", 256, 256,           "M", 512, 512,           "M", 512, 512,           "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256,           "M", 512, 512,           "M", 512, 512,           "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256,      "M", 512, 512, 512,      "M", 512, 512, 512,      "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}
# fmt: on

"""
vgg11:
    num_blocks: [1, 1, 2, 2, 2]
vgg13:
    num_blocks: [2, 2, 2, 2, 2]
vgg16:
    num_blocks: [2, 2, 3, 3, 3]
vgg19:
    num_blocks: [2, 2, 4, 4, 4]
"""

"""
class VGG(nn.Module):
    def __init__(self, config: dict, dataset: ImageDataset):
        super(VGG, self).__init__()

        num_blocks = config["num_blocks"]
        dropout_std = config["dropout"]["std"]

        self.layer1 = self._make_layer(dropout_std[0], num_blocks[0], 3, 64)
        self.layer2 = self._make_layer(dropout_std[1], num_blocks[1], 64, 128)
        self.layer3 = self._make_layer(dropout_std[2], num_blocks[2], 128, 256)
        self.layer4 = self._make_layer(dropout_std[3], num_blocks[3], 256, 512)
        self.layer5 = self._make_layer(dropout_std[4], num_blocks[4], 512, 512)
        self.ave_pool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.classifier = nn.Linear(512, dataset.num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.ave_pool(out)
        out = self.classifier(out)
        return out

    def _make_layers(
        self,
        dropout_std,
        num_blocks,
        in_planes,
        planes,
    ):
        layers = []
        for x in cfg:
            if x == "M":
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)
"""
