import argparse
import json
import random
import itertools
import functools
import torch.multiprocessing as mp

from ml.engine import Task


def get_samples(text):
    for n in text.split(","):
        if "-" in n:
            b, e = n.split("-")
            for i in range(int(b), int(e) + 1):
                yield i
        else:
            yield int(n)


def get_configs(plan_path, group):
    with open(plan_path) as file:
        plan = json.load(file)
    with open(plan["base"]) as file:
        base = json.load(file)

    dims = []
    for factor, value in plan["groups"][group].items():
        dim = plan["factors"][factor]
        if isinstance(value, str):
            dims.append([(value, dim[value])])
        elif isinstance(value, list):
            dim = {k: dim[k] for k in value}
            dims.append(dim.items())
        elif value is True:
            dims.append(dim.items())

    for perm in itertools.product(*dims):
        keys, modifies = zip(*perm)
        label = "_".join(keys)
        config = functools.reduce(modify, modifies, base)
        yield label, config


def modify(x, y):
    if isinstance(x, dict) and isinstance(y, dict):
        x = dict(x)
        for k, v in y.items():
            x[k] = modify(x.get(k), v)
        return x
    return y


def worker(task):
    task.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="configs/base.json")
    parser.add_argument("--plan", "-p", type=str, default="configs/plan.json")
    parser.add_argument("--outputs", "-o", type=str, default="outputs")
    parser.add_argument("--groups", "-g", type=str, default="")
    parser.add_argument("--label", "-l", type=str, default="test")
    parser.add_argument("--device", "-d", type=str, default="cpu")
    parser.add_argument("--checkpoint", "-k", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--samples", "-s", type=str, default="0")
    parser.add_argument("--threads", "-t", type=int, default=1)
    args = parser.parse_args()

    tasks = []

    def add_tasks(config, label):
        for sample in get_samples(args.samples):
            seed = random.getrandbits(31)
            label_ = f"{label}_{sample}"
            path = f"{args.outputs}/{label_}"
            task = Task(
                config=config,
                seed=seed,
                device=args.device,
                verbose=args.verbose,
                label=label_,
                path=path,
                checkpoint=args.checkpoint,
            )
            tasks.append(task)

    if args.groups == "":
        with open(args.config) as file:
            config = json.load(file)
        add_tasks(config, args.label)
    else:
        for group in args.groups.split(","):
            for label, config in get_configs(args.plan, args.groups):
                add_tasks(config, label)

    random.shuffle(tasks)
    mp.set_start_method("spawn")
    with mp.Pool(args.threads) as pool:
        pool.map(worker, tasks)
