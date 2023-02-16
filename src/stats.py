import argparse
import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from engine import Engine
from modules import Capture


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


def do_stat(engine: Engine, train):

    engine.model.eval()
    engine.metrics.reset()

    for inputs, targets in engine.dataloader(train=train):
        outputs = engine.model(inputs)
        add_stat(engine, outputs, targets)

    return engine.metrics.get()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "-d", type=str, default="cpu")
    args = parser.parse_args()

    # for all labels
    labels = os.listdir("outputs")
    engine = None
    for label in labels:
    
        # load config
        with open(f"outputs/{label}/config.json") as file:
            config = json.load(file)
        config["fit"]["num_samples"] = 1
        config["fit"]["loss_fn"]["name"] = "ce"

        # for all checkpoints
        size = config["fit"]["num_epochs"]
        step = config["fit"]["checkpoint_interval"]
        epochs = (10, 20, 50, 100) #range(step, size+1, step)
        for epoch in epochs:

            # skip conditions
            ckpt_path = f"outputs/{label}/checkpoint-{epoch}.pt"
            if not os.path.exists(ckpt_path):
                continue
            stat_path = f"outputs/{label}/statistics-{epoch}.pt"
            if os.path.exists(stat_path):
                continue

            # build engine
            if not engine or label != engine.label:
                seed = random.getrandbits(31)
                engine = Engine(config, seed, label, args.device, checkpoint=False, verbose=True)
                engine.build()

            # load state dict
            state_dict = torch.load(ckpt_path, map_location=args.device)
            engine.model.load_state_dict(state_dict)

            # calculation
            stat = {}
            for k, v in do_stat(engine, train=True).items():
                stat[f"train.{k}"] = v
            for k, v in do_stat(engine, train=False).items():
                stat[f"test.{k}"] = v
            torch.save(stat, stat_path)
            print(stat_path)
