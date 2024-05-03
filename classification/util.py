import numpy as np
import torch
import random
import os


def set_seed(seed: int = 42) -> None:
    """ Set random seed """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_rng(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    set_seed(seed)
    return g


        



