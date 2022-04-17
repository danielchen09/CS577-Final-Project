from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import f1_score

import config
from datasets import *
from models import *
from training import *


def train_sentiment(ds):
    ds_train, ds_val = split_dataset(ds, ratio=0.9)
    train_loader = DataLoader(ds_train, batch_size=config.BATCH_SIZE)
    val_loader = DataLoader(ds_val, batch_size=config.BATCH_SIZE)
    model = NBoW(len(ds.vocabulary), config.EMBEDDING_DIM, len(ds.labels)).to(config.DEVICE)
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    def metric_fn(y_pred, y):
        return f1_score(y.argmax(dim=1).cpu(), y_pred.argmax(dim=1).cpu(), average='micro')

    train(config.TRAIN_EPOCHS, model, optimizer, train_loader, val_loader, loss_fn, metric_fn)


if __name__ == '__main__':
    ds = Hw1Dataset('./data/hw1_sentiment.csv', size=500)
    train_sentiment(ds)