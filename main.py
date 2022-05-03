from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import f1_score

import config
from datasets import *
from models import *
from training import *
from augmentations import *


def train_sentiment(ds_train, ds_val):
    train_loader = DataLoader(ds_train, batch_size=config.BATCH_SIZE)
    val_loader = DataLoader(ds_val, batch_size=len(ds_val) // 2)
    model = LSTMModel(len(ds_train.vocabulary), config.EMBEDDING_DIM, len(ds_train.labels)).to(config.DEVICE)
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    def metric_fn(y_pred, y):
        return f1_score(y.argmax(dim=1).cpu(), y_pred.argmax(dim=1).cpu(), average='micro')

    train(config.TRAIN_EPOCHS, model, optimizer, train_loader, val_loader, loss_fn, metric_fn)


def dataset_creator(ds_class, path):
    def create_dataset(*args, **kwargs):
        return ds_class(path, *args, **kwargs)
    return create_dataset


def main():
    create_dataset = dataset_creator(Hw1Dataset, './data/hw1_sentiment.csv')

    ds = create_dataset()

    augmentations = [BackTranslation()] # easyaug: 0.1667

    train_idx, val_idx = split_indices(ds, 0.005)
    ds_train = create_dataset(vocabulary_set=ds.get_vocabulary_set(), subset=train_idx, augmentations=augmentations)
    ds_val = create_dataset(vocabulary_set=ds.get_vocabulary_set(), subset=val_idx)
    train_sentiment(ds_train, ds_val)


def test():
    create_dataset = dataset_creator(Hw1Dataset, './data/hw1_sentiment.csv')

    ds = create_dataset()

    augmentations = [BackTranslation()]  # easyaug: 0.1667

    i = 5
    with_aug = create_dataset(vocabulary_set=ds.get_vocabulary_set(), subset=(i, i + 1), augmentations=augmentations)
    no_aug = create_dataset(vocabulary_set=ds.get_vocabulary_set(), subset=(i, i + 1))
    print(idx_to_sentence(no_aug[0], ds.idx2word))
    print(idx_to_sentence(with_aug[0], ds.idx2word))


if __name__ == '__main__':
    main()