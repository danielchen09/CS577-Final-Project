import random

import torch
from torch.utils.data import random_split
import torch.nn.functional as F

import config


def get_vocabulary(data, start_idx=1):
    idx = start_idx
    vocabulary = {}
    idx2word = {}
    for sentence in data:
        for word in sentence:
            if word not in vocabulary:
                vocabulary[word] = idx
                idx2word[idx] = word
                idx += 1
    return vocabulary, idx2word


def sentence_to_idx(data, vocabulary):
    idx = []
    for sentence in data:
        row = []
        for word in sentence:
            row.append(vocabulary[word])
        idx.append(row)
    return torch.tensor(idx)


def idx_to_sentence(data, idx2word):
    sentences = []
    for sentence in data:
        row = []
        for word in sentence:
            row.append(idx2word[word.item()])
        sentences.append(row)
    return sentences


def one_hot(data, vocabulary):
    ret = []
    for sentence in data:
        row = []
        for label in sentence:
            row.append(F.one_hot(label, num_classes=len(vocabulary)).type(torch.float32))
        ret.append(row)
    return ret


def split_dataset(ds, ratio=0.9):
    n = len(ds)
    front_size = int(n * ratio)
    return random_split(ds, [front_size, n - front_size])


def not_stopword_indices(x):
    idxs = []
    for idx, word in enumerate(x):
        if word not in config.wordnet.stopwords:
            idxs.append(idx)
    return idxs

