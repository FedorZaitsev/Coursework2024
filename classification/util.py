import numpy as np
import json
import torch
import torch.nn as nn
import random
import os
import albumentations as A
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from models import ConvModel, Swin, QuantizableResNet18, QuantizedResNet18

dict_models = {
    'ConvModel': ConvModel,
    'Swin': Swin,
    'ResNet18': QuantizableResNet18,
    'QuantizedResNet18': QuantizedResNet18
}

def get_model(model_name):
    return dict_models[model_name]

dict_optimizers = {
    'Adam': torch.optim.Adam,
    'AdamW': torch.optim.AdamW,
    'SGD': torch.optim.SGD,
}

dict_schedulers = {
    'LambdaLR': torch.optim.lr_scheduler.LambdaLR,
    'StepLR': torch.optim.lr_scheduler.StepLR,
    'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR,
    'CosineAnnealingLR': torch.optim.lr_scheduler.CosineAnnealingLR,
}

dict_loss_fns = {
    'CrossEntropyLoss': nn.CrossEntropyLoss,
}

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
    gen = torch.Generator()
    gen.manual_seed(seed)
    set_seed(seed)
    return gen


def parse_config(cfg_path):

    data = json.load(open(cfg_path))
    cfg = {}

    cfg['title'] = data['title']

    cfg['seed'] = data['seed']

    aug_cfg = {
        'transforms': [],
        'valid_transforms': [],
        'use_original': True
        }

    for i in range(len(data['aug_cfg']['transforms'])):
        transform = A.from_dict(data['aug_cfg']['transforms'][str(i)])
        aug_cfg['transforms'].append(transform)
    
    for i in range(len(data['aug_cfg']['valid_transforms'])):
        transform = A.from_dict(data['aug_cfg']['valid_transforms'][str(i)])
        aug_cfg['valid_transforms'].append(transform)

    aug_cfg['use_original'] = data['aug_cfg']['use_original']

    cfg['aug_cfg'] = aug_cfg

    cfg['train_ratio'] = data['train_ratio']

    model = {}
    model['instance'] = dict_models[data['model']['instance']]
    model['parameters'] = data['model']['parameters']
    cfg['model'] = model

    cfg['model_type'] = data['model']['instance']

    cfg['num_epochs'] = data['num_epochs']

    optimizer = {}
    optimizer['instance'] = dict_optimizers[data['optimizer']['instance']]
    optimizer['parameters'] = data['optimizer']['parameters']
    cfg['optimizer'] = optimizer

    scheduler = {}
    scheduler['instance'] = dict_schedulers[data['scheduler']['instance']]
    scheduler['parameters'] = data['scheduler']['parameters']
    cfg['scheduler'] = scheduler

    loss_fn = {}
    loss_fn['instance'] = dict_loss_fns[data['loss_fn']['instance']]
    loss_fn['parameters'] = data['loss_fn']['parameters']
    cfg['loss_fn'] = loss_fn

    cfg['model_save_path'] = data['model_save_path']

    cfg['quantize'] = data['quantize']

    cfg['q_model_save_path'] = data['q_model_save_path']

    return cfg