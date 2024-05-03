import numpy as np
import torch
import random
import os

from .img_clf_dataset import create_dataloader
from .img_clf_train_pipeline import clf_train, evaluate


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


def train_model(model, train_dir, aug_cfg, optimizer, loss_fn, 
                scheduler=None, num_epochs=30, valid_dir=None, train_ratio=1, seed=42, 
                title='', wandb_log=False, key=None, proj_name="Coursework2024", verbose=True):
    g = torch.Generator()
    g.manual_seed(seed)
    set_seed(seed)

    train_loader, valid_loader = create_dataloader(cur_dir=train_dir, 
                                               train_ratio=train_ratio, 
                                               train_transforms=aug_cfg['transforms'],
                                               valid_transforms=aug_cfg['valid_transforms'],
                                               use_original=aug_cfg['use_original'],
                                               )

    if valid_dir:
        valid_loader, _ = create_dataloader(cur_dir=valid_dir, 
                                               train_ratio=1, 
                                               train_transforms=aug_cfg['valid_transforms'],
                                               valid_transforms=[],
                                               use_original=aug_cfg['use_original'],
                                               )
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    return clf_train(model, num_epochs, title, train_loader=train_loader, 
                        valid_loader=valid_loader, optimizer=optimizer, loss_fn=loss_fn, scheduler=scheduler, 
                        wandb_log=wandb_log, key=key, proj_name=proj_name, verbose=verbose)
        



