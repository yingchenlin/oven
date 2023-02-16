import os
import json
import numpy as np
import torch
import logging
from tqdm import tqdm

if __package__ == "":
    from datasets import get_dataset
    from modules import get_model, get_loss_fn, get_optim
    from metrics import get_metrics
else:
    from .datasets import get_dataset
    from .modules import get_model, get_loss_fn, get_optim
    from .metrics import get_metrics


class Engine:

    def __init__(self, config, seed, label, device, checkpoint, verbose):
        self.config = config
        self.seed = seed
        self.label = label
        self.device = device
        self.checkpoint = checkpoint
        self.verbose = verbose

        self.base_path = f"outputs/{self.label}"
        self.num_epochs = self.config["fit"]["num_epochs"]
        self.num_samples = self.config["fit"]["num_samples"]
        self.checkpoint_interval = self.config["fit"]["checkpoint_interval"]
        self.epoch = 0
        self.logs = []
    
    @property
    def checkpointing(self):
        return self.checkpoint and self.epoch % self.checkpoint_interval == 0

    def run(self):
        if os.path.exists(f"{self.base_path}/done"):
            return
        self.build()
        self.prepare()
        while self.epoch < self.num_epochs:
            self.epoch += 1
            train_metrics = self.train(self.dataloader(train=True))
            test_metrics = self.eval(self.dataloader(train=False))
            self.log(train=train_metrics, test=test_metrics)
        with open(f"{self.base_path}/done", "w"):
            pass

    def build(self):

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

    def prepare(self):

        os.makedirs(self.base_path, exist_ok=True)
        with open(f"{self.base_path}/config.json", "w") as file:
            json.dump(self.config, file, indent=2)

        logging.basicConfig(
            format='[%(asctime)s] %(message)s',
            datefmt='%H:%M:%S',
            level=logging.INFO)

        logging.info(f"{self.label} config={self.config}")
        logging.info(f"{self.label} model={self.model}")

    def dataloader(self, train):

        dataloader = (
            self.dataset.train_dataloader if train else 
            self.dataset.test_dataloader)

        if self.verbose:
            dataloader = tqdm(dataloader, leave=False)

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device).expand(self.num_samples, *inputs.shape)
            targets = targets.to(self.device).expand(self.num_samples, *targets.shape)
            yield inputs, targets

    def train(self, dataloader):

        self.model.train()
        self.loss_fn.train()

        self.metrics.reset()

        for inputs, targets in dataloader:

            self.optim.zero_grad()
            outputs = self.model(inputs)
            losses = self.loss_fn(inputs, outputs, targets, self.model)
            losses.mean().backward()
            self.optim.step()

            self.metrics.add_losses(losses)
            self.metrics.add_ranks(outputs, targets)
            self.metrics.add_states(self.model)

        return self.metrics.get()

    def eval(self, dataloader):

        self.model.eval()
        self.loss_fn.eval()

        self.metrics.reset()
        self.metrics.add_params(self.model)

        for inputs, targets in dataloader:

            self.model.zero_grad()
            outputs = self.model(inputs)
            losses = self.loss_fn(inputs, outputs, targets, self.model)
            losses.mean().backward()

            self.metrics.add_losses(losses)
            self.metrics.add_ranks(outputs, targets)
            self.metrics.add_states(self.model)

        return self.metrics.get()

    def log(self, **kwargs):

        scalars = {"epoch": self.epoch}
        for phase, metrics in kwargs.items():
            for k, v in metrics.items():
                scalars[f"{phase}.{k}"] = v

        self.logs.append(scalars)
        with open(f"{self.base_path}/logs.json", "w") as file:
            json.dump(self.logs, file, indent=2)

        if self.checkpointing:
            path = f"{self.base_path}/checkpoint-{self.epoch}.pt"
            torch.save(self.model.state_dict(), path)

        log_str = ""
        for k, v in scalars.items():
            if "$" in k or not isinstance(v, (int, float)): continue
            if isinstance(v, float): v = f"{v:.4f}"
            log_str += f"{k}={v} "
        logging.info(f"{self.label} {log_str[:-1]}")
