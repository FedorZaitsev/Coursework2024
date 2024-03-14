import os
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import v2
from tqdm import tqdm
from IPython.display import clear_output
import wandb


def train(model, train_loader, optimizer, loss_fn) -> float:
    model.train()

    device = next(model.parameters()).device

    train_loss = 0
    total = 0
    correct = 0

    for x, y in tqdm(train_loader, desc='Train'):

        x = x.to(device)        
        y = y.type(torch.LongTensor)
        y = y.to(device)

        
        optimizer.zero_grad()
                
        output = model(x)
        output.to(device)

        loss = loss_fn(output, y)

        train_loss += loss.item()

        loss.backward()

        optimizer.step()

        _, y_pred = output.max(dim=1)
        total += y.size(0)
        correct += (y == y_pred).sum().item()

    train_loss /= len(train_loader)
    accuracy = correct / total

    return train_loss, accuracy


@torch.inference_mode()
def evaluate(model, loader, optimizer, loss_fn):
    model.eval()

    device = next(model.parameters()).device

    total_loss = 0
    total = 0
    correct = 0

    for x, y in tqdm(loader, desc='Evaluation'):
        
        x = x.to(device)        
        y = y.type(torch.LongTensor)
        y = y.to(device)
        
        output = model(x)
        output.to(device)

        loss = loss_fn(output, y)

        total_loss += loss.item()

        _, y_pred = torch.max(output, 1)
        total += y.size(0)
        correct += (y_pred == y).sum().item()

    total_loss /= len(loader)
    accuracy = correct / total

    return total_loss, accuracy


def plot_stats(
    train_loss,
    valid_loss,
    train_accuracy,
    valid_accuracy,
    title
):
    plt.figure(figsize=(16, 8))

    plt.title(title + ' loss')

    plt.plot(train_loss, label='Train loss')
    plt.plot(valid_loss, label='Valid loss')
    plt.legend()
    plt.grid()

    plt.show()

    plt.figure(figsize=(16, 8))

    plt.title(title + ' accuracy')

    plt.plot(train_accuracy, label='Train accuracy')
    plt.plot(valid_accuracy, label='Valid accuracy')
    plt.legend()
    plt.grid()

    plt.show()


def whole_train_valid_cycle(model, num_epochs, title, train_loader, valid_loader, optimizer, loss_fn, silent=True):
    train_loss_history, valid_loss_history = [], []
    train_acc_history, valid_acc_history = [], []

    if not silent:
        run = wandb.init(
            # Set the project where this run will be logged
            project="my-awesome-project",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": optimizer.param_groups[-1]['lr'],
                "epochs": num_epochs,
                "vocab_size": model.vocab_size_de
            },
        )

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_loader, optimizer, loss_fn)
        valid_loss, valid_accuracy = evaluate(model, valid_loader, optimizer, loss_fn)

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)
        
        train_acc_history.append(train_accuracy)
        valid_acc_history.append(valid_accuracy)

        if not silent:
            wandb.log({"train_loss": train_loss, "train_acc": train_accuracy, "valid_loss": valid_loss, "val_acc": valid_accuracy})

        clear_output()

        plot_stats(
            train_loss_history, valid_loss_history,
            train_acc_history, valid_acc_history,
            title
        )
