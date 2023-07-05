import os
import json
import numpy as np
import torch
import logging
from tqdm import tqdm
from torch import Tensor
from typing import Dict, Iterator, Tuple

from ml.datasets import get_dataset
from ml.modules import get_model, get_loss_fn, get_optim
from ml.metrics import get_metrics


Batch = Tuple[Tensor, Tensor]


class Engine:
    def __init__(
        self,
        config: dict,
        seed: int,
        label: str,
        path: str,
        device: str,
        checkpoint: bool,
        verbose: bool,
    ) -> None:
        self.config = config
        self.seed = seed
        self.label = label
        self.path = path
        self.device = device
        self.checkpoint = checkpoint
        self.verbose = verbose

        self.num_epochs: int = self.config["fit"]["num_epochs"]
        self.num_samples: int = self.config["fit"]["num_samples"]
        self.checkpoint_interval: int = self.config["fit"]["checkpoint_interval"]
        self.epoch: int = 0
        self.logs: list = []

    @property
    def checkpointing(self) -> bool:
        interval = self.checkpoint_interval
        return interval != 0 and self.epoch % interval == 0

    def run(self) -> None:
        if os.path.exists(f"{self.path}/done"):
            return
        self.build()
        self.prepare()
        while self.epoch < self.num_epochs:
            self.epoch += 1
            train_metrics = self.train(self.iterator(train=True))
            test_metrics = self.eval(self.iterator(train=False))
            self.log(train=train_metrics, test=test_metrics)
            if np.isnan(train_metrics["loss"]):
                break
        with open(f"{self.path}/done", "w"):
            pass

    def build(self) -> None:
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        self.dataset = get_dataset(self.config["dataset"])
        self.model = get_model(self.config["model"], self.dataset)
        self.optim = get_optim(self.config["fit"]["optim"], self.model)
        self.loss_fn = get_loss_fn(self.config["fit"]["loss_fn"])
        self.metrics = get_metrics(self.config["metrics"])

        self.model.to(self.device)
        self.loss_fn.to(self.device)

    def prepare(self) -> None:
        os.makedirs(self.path, exist_ok=True)
        with open(f"{self.path}/config.json", "w") as file:
            json.dump(self.config, file, indent=2)

        logging.basicConfig(
            format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S", level=logging.INFO
        )

        logging.info(f"{self.label} config={self.config}")
        logging.info(f"{self.label} model={self.model}")

    def iterator(self, train: bool) -> Iterator[Batch]:
        dataloader = (
            self.dataset.train_dataloader if train else self.dataset.test_dataloader
        )

        if self.verbose:
            dataloader = tqdm(dataloader, leave=False)

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            inputs = inputs.expand(self.num_samples, *inputs.shape)
            targets = targets.expand(self.num_samples, *targets.shape)
            yield inputs, targets

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
        # losses = losses + self.model.reg_loss(outputs)

        return losses

    def train(self, iterator: Iterator[Batch]) -> Dict[str, float]:
        self.model.train()
        self.loss_fn.train()

        self.metrics.reset()

        for inputs, targets in iterator:
            self.optim.zero_grad()
            outputs = self.model(inputs)
            losses = self.loss(inputs, outputs, targets)
            losses.mean().backward()
            self.optim.step()

            self.metrics.add_losses(losses)
            self.metrics.add_ranks(outputs, targets)
            self.metrics.add_states(self.model)

        return self.metrics.get()

    def eval(self, iterator: Iterator[Batch]) -> dict:
        self.model.eval()
        self.loss_fn.eval()

        self.metrics.reset()
        self.metrics.add_params(self.model)

        for inputs, targets in iterator:
            self.model.zero_grad()
            outputs = self.model(inputs)
            losses = self.loss(inputs, outputs, targets)
            losses.mean().backward()

            self.metrics.add_losses(losses)
            self.metrics.add_ranks(outputs, targets)
            self.metrics.add_states(self.model)

        return self.metrics.get()

    def log(self, **kwargs) -> None:
        scalars = {"epoch": self.epoch}
        for phase, metrics in kwargs.items():
            for k, v in metrics.items():
                scalars[f"{phase}.{k}"] = v

        self.logs.append(scalars)
        with open(f"{self.path}/logs.json", "w") as file:
            json.dump(self.logs, file, indent=2)

        if self.checkpointing:
            path = f"{self.path}/checkpoint-{self.epoch}.pt"
            torch.save(self.model.state_dict(), path)

        log_str = ""
        for k, v in scalars.items():
            if "$" in k or not isinstance(v, (int, float)):
                continue
            if isinstance(v, float):
                v = f"{v:.4f}"
            log_str += f"{k}={v} "
        logging.info(f"{self.label} {log_str[:-1]}")
