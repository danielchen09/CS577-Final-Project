import torch


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