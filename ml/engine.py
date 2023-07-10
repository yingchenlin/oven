import os
import json
import numpy as np
import torch
import logging
from tqdm import tqdm
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Dict, Iterator, Tuple

from ml.datasets import get_dataset
from ml.modules import get_model, get_loss_fn, get_optim
from ml.metrics import get_metrics


class Engine:
    def __init__(
        self,
        config: dict,
        seed: int = 0,
        device: str = "cpu",
        verbose: bool = False,
    ) -> None:
        self.seed = seed
        self.device = device
        self.verbose = verbose

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        self.dataset = get_dataset(config["dataset"])
        self.model = get_model(config["model"], self.dataset)
        self.optim = get_optim(config["fit"]["optim"], self.model)
        self.loss_fn = get_loss_fn(config["fit"]["loss_fn"])
        self.metrics = get_metrics(config["metrics"])

        self.num_samples: int = config["fit"]["num_samples"]

        self.model.to(self.device)
        self.loss_fn.to(self.device)

    def load(self, path: str) -> None:
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)

    def save(self, path: str) -> None:
        state_dict = self.model.state_dict()
        torch.save(state_dict, path)

    def train(self) -> Dict[str, float]:
        self.model.train()
        self.loss_fn.train()

        self.metrics.reset()

        dataloader = self.dataset.train_dataloader
        if self.verbose:
            dataloader = tqdm(dataloader, leave=False)

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device).expand(self.num_samples, *inputs.shape)
            targets = targets.to(self.device).expand(self.num_samples, *targets.shape)

            self.optim.zero_grad()
            outputs = self.model(inputs)
            losses = self.loss(inputs, outputs, targets)
            losses.mean().backward()
            self.optim.step()

            self.metrics.add_losses(losses)
            self.metrics.add_ranks(outputs, targets)
            self.metrics.add_states(self.model)

        return self.metrics.get()

    def eval(self) -> Dict[str, float]:
        self.model.eval()
        self.loss_fn.eval()

        self.metrics.reset()
        self.metrics.add_params(self.model)

        dataloader = self.dataset.test_dataloader
        if self.verbose:
            dataloader = tqdm(dataloader, leave=False)

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device).expand(self.num_samples, *inputs.shape)
            targets = targets.to(self.device).expand(self.num_samples, *targets.shape)

            self.model.zero_grad()
            outputs = self.model(inputs)
            losses = self.loss(inputs, outputs, targets)
            losses.mean().backward()

            self.metrics.add_losses(losses)
            self.metrics.add_ranks(outputs, targets)
            self.metrics.add_states(self.model)

        return self.metrics.get()

    def loss(self, inputs: Tensor, outputs: Tensor, targets: Tensor) -> Tensor:
        """
        if isinstance(self.loss_fn, DiffLoss):
            if self.model.training:
                self.model.eval()
                mean = self.model(inputs)
                diff = outputs - mean
                self.model.train()
            else:
                mean = outputs
                diff = torch.zeros_like(mean)
            outputs = (mean, diff)
        """
        losses = self.loss_fn(outputs, targets)
        losses = losses + self.model.reg_loss(outputs)
        return losses


class Task:
    engine: Engine

    def __init__(
        self,
        config: dict,
        seed: int = 0,
        device: str = "cpu",
        verbose: bool = False,
        label: str = "",
        path: str = "",
        checkpoint: bool = False,
    ):
        self.config = config
        self.seed = seed
        self.device = device
        self.verbose = verbose

        self.label = label
        self.path = path
        self.checkpoint = checkpoint

        self.num_epochs: int = self.config["fit"]["num_epochs"]
        self.checkpoint_interval: int = self.config["fit"]["checkpoint_interval"]
        self.epoch: int = 0
        self.records: list = []

    def run(self) -> None:
        if os.path.exists(f"{self.path}/done"):
            return

        os.makedirs(self.path, exist_ok=True)
        with open(f"{self.path}/config.json", "w") as file:
            json.dump(self.config, file, indent=2)

        logging.basicConfig(
            format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S", level=logging.INFO
        )

        self.engine = Engine(self.config, self.seed, self.device, self.verbose)
        logging.info(f"{self.label} config={self.config}")
        logging.info(f"{self.label} model={self.engine.model}")
        logging.info(f"{self.label} loss_fn={self.engine.loss_fn}")

        while self.epoch < self.num_epochs:
            self.epoch += 1

            train_metrics = self.engine.train()
            test_metrics = self.engine.eval()

            self.output(train_metrics, test_metrics)

            interval = self.checkpoint_interval
            if interval != 0 and self.epoch % interval == 0:
                self.engine.save(f"{self.path}/checkpoint-{self.epoch}.pt")

            if np.isnan(train_metrics["loss"]):
                break

        with open(f"{self.path}/done", "w"):
            pass

    def output(self, train_metrics, test_metrics) -> None:
        # build record
        record = {"epoch": self.epoch}
        for k, v in train_metrics.items():
            record[f"train.{k}"] = v
        for k, v in test_metrics.items():
            record[f"test.{k}"] = v

        # print
        log_str = ""
        for k, v in record.items():
            if "$" in k or not isinstance(v, (int, float)):
                continue
            if isinstance(v, float):
                v = f"{v:.4f}"
            log_str += f"{k}={v} "
        logging.info(f"{self.label} {log_str[:-1]}")

        # file system
        self.records.append(record)
        with open(f"{self.path}/logs.json", "w") as file:
            json.dump(self.records, file, indent=2)
