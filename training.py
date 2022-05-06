import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

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
    epoch_number = []
    eval_metric = []
    train_accuracy = []
    for epoch in range(epochs):
        model.train()
        for x, y in tqdm(train_loader, desc=f'epoch {epoch + 1}/{epochs}'):
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            optimizer.zero_grad()
            y_pred = model(x)
            model_loss = loss_fn(y_pred, y)
            model_loss.backward()
            optimizer.step()

        if epoch % config.EVAL_EPOCH == 0:
            val_loss, val_metric = evaluate(model, val_loader, loss_fn, metric_fn)
            train_loss, train_metric = evaluate(model, train_loader, loss_fn, metric_fn)

            print('\n')
            print(f'validation metric at epoch {epoch}: {val_metric}')
            print(f'validation loss at epoch {epoch}: {val_loss}')
            epoch_number.append(epoch)
            eval_metric.append(val_metric)
            train_accuracy.append(train_metric)

    fig, ax = plt.subplots()
    ax.plot(epoch_number, eval_metric, label="validation accuracy")
    ax.plot(epoch_number, train_accuracy, label="training accuracy")
    leg = ax.legend()
    plt.xlabel("epoch #")
    plt.ylabel("accuracy")
    plt.savefig("graph.png")
    plt.show()