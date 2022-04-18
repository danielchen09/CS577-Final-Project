from torch.utils.data import Dataset
import gzip
import ast
import csv

import config
from utils import *
from augmentations import augment_sentences


class SentimentDataset(Dataset):
    def __init__(self, filename, size=-1, augmentations=None):
        if augmentations is None:
            augmentations = []
        self.filename = filename
        self.train_data, self.train_label = self.parse()
        if size != -1:
            self.train_data = self.train_data[:size]
            self.train_label = self.train_label[:size]
        augment_sentences(self.train_data, self.train_label, augmentations)
        self.vocabulary, self.idx2word = get_vocabulary(self.train_data, start_idx=1)
        self.vocabulary[config.PAD] = 0
        self.idx2word[0] = config.PAD

        max_len = max([len(x) for x in self.train_data])
        for i in range(len(self.train_data)):
            for _ in range(max_len - len(self.train_data[i])):
                self.train_data[i].append(config.PAD)

        self.train_data = sentence_to_idx(self.train_data, self.vocabulary)
        self.labels, self.idx2label = get_vocabulary(self.train_label, start_idx=0)
        self.train_label = sentence_to_idx(self.train_label, self.labels)
        self.train_label = one_hot(self.train_label, self.labels)

    def parse(self):
        return None, None

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item):
        return self.train_data[item], self.train_label[item][0]


class AmazonReviewDataset(SentimentDataset):
    def parse(self):
        reviews = gzip.open(self.filename, 'r')
        x = []
        y = []
        for review in reviews:
            review = review.decode('utf-8')
            review = review.replace('true', 'True')
            review = review.replace('false', 'False')
            data = ast.literal_eval(review)
            if 'reviewText' not in data:
                continue
            x.append(data['reviewText'].lower().split())
            y.append([data['overall']])
        return x, y


class Hw1Dataset(SentimentDataset):
    def parse(self):
        x = []
        y = []
        with open(self.filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                x.append(row['text'].lower().split())
                y.append([row['emotions']])
        return x, y
