from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ml.datasets.utils import *

import copy
from functools import lru_cache


class ImageDataset:
    input_shape: Shape
    num_classes: int

    def __init__(self, config: dict, path: str) -> None:
        self.train_dataset = self._get_dataset(config, path, True)
        self.test_dataset = self._get_dataset(config, path, False)
        self.train_dataloader = self._get_dataloader(
            config["train"], self.train_dataset
        )
        self.test_dataloader = self._get_dataloader(config["test"], self.test_dataset)

    @classmethod
    def _get_dataset(
        cls, config: dict, path: str, train: bool
    ) -> datasets.VisionDataset:
        dataset = cls._load_dataset(path, train)
        dataset = copy.copy(dataset)
        dataset.transform = get_transform(config["transform"], cls.input_shape, train)
        return dataset

    @classmethod
    def _get_dataloader(
        cls, config: dict, dataset: datasets.VisionDataset
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=config["shuffle"],
        )

    @classmethod
    def _load_dataset(cls, path: str, train: bool) -> datasets.VisionDataset:
        raise NotImplementedError(cls.__name__)


def get_normalize(config: dict, input_shape: Shape) -> transforms.Normalize:
    return transforms.Normalize(
        [config["mean"]] * input_shape[0],
        [config["std"]] * input_shape[0],
    )


def get_transform(
    config: dict, input_shape: Shape, training: bool
) -> transforms.Compose:
    layers = []
    if training:
        if "flip" in config:
            layers.append(transforms.RandomHorizontalFlip(**config["flip"]))
        if "crop" in config:
            layers.append(transforms.RandomCrop(size=input_shape[1:], **config["crop"]))
        if "erase" in config:
            layers.append(transforms.RandomErasing(**config["erase"]))
        if "rotate" in config:
            layers.append(transforms.RandomRotation(**config["rotate"]))
        if "affine" in config:
            layers.append(transforms.RandomAffine(**config["affine"]))
        if "color" in config:
            layers.append(transforms.ColorJitter(**config["color"]))
    layers.append(transforms.ToTensor())
    layers.append(get_normalize(config["normalize"], input_shape))
    return transforms.Compose(layers)


class MNIST(ImageDataset):
    input_shape = (1, 32, 32)
    num_classes = 10

    @classmethod
    @lru_cache
    def _load_dataset(cls, path: str, train: bool) -> datasets.VisionDataset:
        return datasets.MNIST(path, train=train, download=True)


class CIFAR10(ImageDataset):
    input_shape = (3, 32, 32)
    num_classes = 10

    @classmethod
    @lru_cache
    def _load_dataset(cls, path: str, train: bool) -> datasets.VisionDataset:
        return datasets.CIFAR10(path, train=train, download=True)


class CIFAR100(ImageDataset):
    input_shape = (3, 32, 32)
    num_classes = 100

    @classmethod
    @lru_cache
    def _load_dataset(cls, path: str, train: bool) -> datasets.VisionDataset:
        return datasets.CIFAR100(path, train=train, download=True)


class SVHN(ImageDataset):
    input_shape = (3, 32, 32)
    num_classes = 10

    @classmethod
    @lru_cache
    def _load_dataset(cls, path: str, train: bool) -> datasets.VisionDataset:
        split = "train" if train else "test"
        return datasets.SVHN(path, split=split, download=True)
