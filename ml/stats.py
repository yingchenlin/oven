import argparse
import os
import json
import random
import torch
import torch.nn.functional as F
import tqdm
from ml.engine import Engine
from ml.models import Capture
from torch import Tensor
from typing import Mapping

"""
def add_stat(engine: Engine, outputs, targets):

    x, i = outputs, targets
    p = x.softmax(-1)
    j0 = p - F.one_hot(i, x.shape[-1])
    h0 = p.diag_embed() - p.unsqueeze(-2) * p.unsqueeze(-1)
    sp = torch.ones_like(p).diag_embed()

    modules = list(engine.model.named_children())
    for name, m in reversed(modules[3:-1]):
        if isinstance(m, nn.Linear):
            sp = sp @ m.weight
        if isinstance(m, nn.ReLU):
            sp = sp * (x > 0).unsqueeze(-2)
        if isinstance(m, Capture):
            x = m.state
            j = torch.einsum("sbi,sbij->sbj", j0, sp)
            h = torch.einsum("sbij,sbik,sbjl->sbkl", h0, sp, sp)
            engine.metrics._agg_moments(f"${name}.state", x)
            engine.metrics._agg_moments(f"${name}.jacob", j)
            engine.metrics._agg_tensor(f"${name}.hess", "m1", h)

"""


def do_scan(path, device):
    # for all labels
    labels = os.listdir(path)
    for label in sorted(labels):
        if "dist" in label:
            continue

        # load config
        with open(f"{path}/{label}/config.json") as file:
            config = json.load(file)

        # for all checkpoints
        """
        size = config["fit"]["num_epochs"]
        step = config["fit"]["checkpoint_interval"]
        epochs = range(step, size + 1, step)
        """
        epochs = (20, 40, 60, 80, 120, 160)
        for epoch in epochs:
            # skip conditions
            ckpt_path = f"{path}/{label}/checkpoint-{epoch}.pt"
            if not os.path.exists(ckpt_path):
                continue
            stat_path = f"{path}/{label}/statistics-{epoch}.pt"
            if os.path.exists(stat_path):
                continue

            # build engine
            seed = random.getrandbits(31)
            engine = Engine(config, seed, device, True)

            # load state dict
            state_dict = torch.load(ckpt_path, map_location=device)
            engine.model.load_state_dict(state_dict)

            # calculation
            stat = {}
            engine.run(train=True, moments=True)
            for k, v in engine.metrics.get_tensors().items():
                stat[f"test.{k}"] = v
            engine.run(train=False, moments=True)
            for k, v in engine.metrics.get_tensors().items():
                stat[f"train.{k}"] = v

            torch.save(stat, stat_path)
            print(stat_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str, default="outputs")
    parser.add_argument("--device", "-d", type=str, default="cpu")
    args = parser.parse_args()

    do_scan(args.path, args.device)
