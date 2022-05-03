import random

import torch
from torch.utils.data import random_split
import torch.nn.functional as F
from tqdm import tqdm
import pickle

import config


def get_vocabulary(data, use_wordnet=True, contains_na=True, save=False):
    idx = 2 if contains_na else 1
    vocabulary = {}
    idx2word = {}
    for sentence in tqdm(data, desc='Creating vocabulary'):
        for word in sentence:
            words_to_add = [word]
            if use_wordnet:
                words_to_add = words_to_add + config.wordnet.get_synonyms(word)
            for word_to_add in words_to_add:
                if word_to_add not in vocabulary:
                    vocabulary[word_to_add] = idx
                    idx2word[idx] = word_to_add
                    idx += 1
    print(f'Vocabulary size: {len(vocabulary)}')
    vocabulary[config.PAD] = 0
    idx2word[0] = config.PAD
    if contains_na:
        vocabulary[config.UNK] = 1
        idx2word[1] = config.UNK
    if save:
        save_pickle((vocabulary, idx2word), 'voc_set.pickle')

    return vocabulary, idx2word


def inverse_vocabulary(vocabulary):
    return {v: k for k, v in vocabulary.items()}


def sentence_to_idx(sentence, vocabulary):
    row = []
    for word in sentence:
        if word in vocabulary:
            row.append(vocabulary[word])
        else:
            row.append(vocabulary[config.UNK])
    return torch.tensor(row)


def idx_to_sentence(data, idx2word):
    sentences = []
    for sentence in data:
        row = []
        for word in sentence:
            row.append(idx2word[word.item()])
        sentences.append(row)
    return sentences


def one_hot(sentence, vocabulary):
    row = []
    for label in sentence:
        row.append(F.one_hot(label, num_classes=len(vocabulary)).type(torch.float32))
    return row


def split_indices(ds, ratio=0.9):
    n = len(ds)
    front_size = int(n * ratio)
    return (0, front_size), (front_size, n)


def not_stopword_indices(x):
    idxs = []
    for idx, word in enumerate(x):
        if word not in config.wordnet.stopwords:
            idxs.append(idx)
    return idxs


def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)