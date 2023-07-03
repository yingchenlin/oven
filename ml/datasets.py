import torch
import torch.utils.data
import torchvision

import copy
import functools
from typing import Tuple


def get_dataset(config):
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
    raise NotImplementedError()


def get_dataloader(config, dataset):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
    )


class Dataset:
    input_shape: Tuple[int, ...]
    num_classes: int

    def __init__(self, config, path):
        train_transform = get_transform(config["transform"], self.input_shape, True)
        test_transform = get_transform(config["transform"], self.input_shape, False)
        self.train_dataset = self._get_dataset(path, True, train_transform)
        self.test_dataset = self._get_dataset(path, False, test_transform)
        self.train_dataloader = get_dataloader(config["train"], self.train_dataset)
        self.test_dataloader = get_dataloader(config["test"], self.test_dataset)

    def _get_dataset(self, path, train, transform):
        dataset = copy.copy(self._load_dataset(path, train))
        dataset.transform = transform
        dataset.transforms = torchvision.datasets.vision.StandardTransform(
            transform, None
        )
        return dataset

    @staticmethod
    def _load_dataset(path, train):
        raise NotImplementedError()


def get_normalize(config, input_shape):
    return torchvision.transforms.Normalize(
        [config["mean"]] * input_shape[0],
        [config["std"]] * input_shape[0],
    )


def get_transform(config, input_shape, training):
    transforms = []
    if training:
        if "flip" in config:
            transforms.append(
                torchvision.transforms.RandomHorizontalFlip(**config["flip"])
            )
        if "crop" in config:
            transforms.append(
                torchvision.transforms.RandomCrop(
                    size=input_shape[1:], **config["crop"]
                )
            )
        if "erase" in config:
            transforms.append(torchvision.transforms.RandomErasing(**config["rotate"]))
        if "rotate" in config:
            transforms.append(torchvision.transforms.RandomRotation(**config["rotate"]))
        if "affine" in config:
            transforms.append(torchvision.transforms.RandomAffine(**config["affine"]))
        if "color" in config:
            transforms.append(torchvision.transforms.ColorJitter(**config["color"]))
    transforms.append(torchvision.transforms.ToTensor())
    transforms.append(get_normalize(config["normalize"], input_shape))
    return torchvision.transforms.Compose(transforms)


class MNIST(Dataset):
    input_shape = (1, 32, 32)
    num_classes = 10

    @staticmethod
    @functools.cache
    def _load_data(path, train):
        return torchvision.datasets.MNIST(path, train=train, download=True)


class CIFAR10(Dataset):
    input_shape = (3, 32, 32)
    num_classes = 10

    @staticmethod
    @functools.cache
    def _load_dataset(path, train):
        return torchvision.datasets.CIFAR10(path, train=train, download=True)


class CIFAR100(Dataset):
    input_shape = (3, 32, 32)
    num_classes = 100

    @staticmethod
    @functools.cache
    def _load_dataset(path, train):
        return torchvision.datasets.CIFAR100(path, train=train, download=True)


class SVHN(Dataset):
    input_shape = (3, 32, 32)
    num_classes = 10

    @staticmethod
    @functools.cache
    def _load_dataset(path, train):
        return torchvision.datasets.SVHN(path, train=train, download=True)
