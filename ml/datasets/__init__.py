from __future__ import annotations

from ml.datasets.image import *


def get_dataset(config: dict) -> ImageDataset:
    name = config["name"]
    path = f"datasets/{name}"
    if name == "mnist":
        return MNIST(config, path)
    if name == "cifar-10":
        return CIFAR10(config, path)
    if name == "cifar-100":
        return CIFAR100(config, path)
    if name == "svhn":
        return SVHN(config, path)
    raise NotImplementedError(name)
