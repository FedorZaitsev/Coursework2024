import torch
import torch.nn as nn
from tqdm import tqdm
import torchmetrics
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
from PIL import Image
import torchvision.transforms as T
import wandb


def train(model, train_loader, optimizer, loss_fn) -> float:
    model.train()

    device = next(model.parameters()).device

    train_loss = 0
    total = 0
    correct = 0
    
    iou_metric = torchmetrics.JaccardIndex(task='multiclass', num_classes=3, ignore_index=2).to(device)

    for x, y in tqdm(train_loader, desc='Train'):
        bs = y.size(0)

        x, y = x.to(device), y.squeeze(1).to(device)

        optimizer.zero_grad()

        output = model(x)

        loss = loss_fn(output.reshape(bs, 3, -1), y.reshape(bs, -1).long())

        train_loss += loss.item()

        loss.backward()

        optimizer.step()

        _, y_pred = output.max(dim=1)
        total += y.size(0) * y.size(1) * y.size(2)
        correct += (y == y_pred).sum().item()
        iou_metric.update(output, y)

    train_loss /= len(train_loader)
    accuracy = correct / total

    return train_loss, accuracy, iou_metric.compute().item()


@torch.inference_mode()
def evaluate(model, loader, optimizer, loss_fn):
    model.eval()

    device = next(model.parameters()).device

    total_loss = 0
    total = 0
    correct = 0
    iou_metric = torchmetrics.JaccardIndex(task='multiclass', num_classes=3, ignore_index=2).to(device)

    for x, y in tqdm(loader, desc='Evaluation'):
        bs = y.size(0)

        x, y = x.to(device), y.squeeze(1).to(device)

        output = model(x)

        loss = loss_fn(output.reshape(bs, 3, -1), y.reshape(bs, -1).long())

        total_loss += loss.item()

        _, y_pred = output.max(dim=1)
        total += y.size(0) * y.size(1) * y.size(2)
        correct += (y == y_pred).sum().item()
        iou_metric.update(output, y)

    total_loss /= len(loader)
    accuracy = correct / total

    return total_loss, accuracy, iou_metric.compute().item()


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


@torch.inference_mode()
def visualize(model, batch, max_vis=3):
    model.eval()

    xs, ys = batch
    
    to_pil = T.ToPILImage()

    for i, (x, y) in enumerate(zip(xs, ys)):
        prediction = model(x.unsqueeze(0).cuda()).squeeze(0).max(dim=0)[1]

        fig, ax = plt.subplots(1, 3, figsize=(24, 8), facecolor='white')

        ax[0].imshow(to_pil(x))
        ax[1].imshow(to_pil(y.to(torch.uint8)))
        ax[2].imshow(to_pil(prediction.to(torch.uint8)))

        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')

        ax[0].set_title('Original image')
        ax[1].set_title('Segmentation mask')
        ax[2].set_title('Prediction')

        plt.subplots_adjust(wspace=0, hspace=0.1)
        plt.show()

        if i >= max_vis:
            break


def seg_train(model, num_epochs, title, train_loader, valid_loader, optimizer, loss_fn, scheduler=None, max_vis=3, silent=True):
    train_loss_history, valid_loss_history = [], []
    train_iou_history, valid_iou_history = [], []

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
        train_loss, train_accuracy, train_iou = train(model, train_loader, optimizer, loss_fn)
        valid_loss, valid_accuracy, valid_iou = evaluate(model, valid_loader, optimizer, loss_fn)

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)

        train_iou_history.append(train_iou)
        valid_iou_history.append(valid_iou)
        
        if not silent:
            wandb.log({"train_loss": train_loss, "train_iou": train_iou, "valid_loss": valid_loss, "val_iou": valid_iou})
        
        if scheduler is not None:
            scheduler.step()

        clear_output()

        plot_stats(
            train_loss_history, valid_loss_history,
            train_iou_history, valid_iou_history,
            title
        )

        visualize(model, next(iter(valid_loader)), max_vis)