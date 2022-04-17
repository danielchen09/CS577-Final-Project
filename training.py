import torch
from tqdm import tqdm
import numpy as np

import config


def evaluate(model, loader, loss_fn, metric_fn):
    model.eval()
    with torch.no_grad():
        metrics = []
        losses = []
        for x, y in loader:
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            y_pred = model(x)
            losses.append(loss_fn(y_pred, y).cpu())
            metrics.append(metric_fn(y_pred, y))
    return np.mean(losses), np.mean(metrics)


def train(epochs, model, optimizer, train_loader, val_loader, loss_fn, metric_fn):
    for epoch in tqdm(range(epochs), desc='training progress'):
        model.train()
        for x, y in train_loader:
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            optimizer.zero_grad()
            y_pred = model(x)
            model_loss = loss_fn(y_pred, y)
            model_loss.backward()
            optimizer.step()

        if (epoch + 1) % config.EVAL_EPOCH == 0:
            val_loss, val_metric = evaluate(model, val_loader, loss_fn, metric_fn)
            print('\n')
            print(f'validation metric at epoch {epoch}: {val_metric}')
            print(f'validation loss at epoch {epoch}: {val_loss}')
