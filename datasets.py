from torch.utils.data import Dataset
import gzip
import ast
import csv
import os

import config
from utils import *
from augmentations import augment_sentences


class SentimentDataset(Dataset):
    def __init__(self, filename, vocabulary_set=None, subset=None, augmentations=None):
        self.augmentations = []
        if augmentations is not None:
            self.augmentations = augmentations
        self.filename = filename
        self.train_data, self.train_label = self.parse()
        if subset is not None:
            self.train_data = self.train_data[subset[0]:subset[1]]
            self.train_label = self.train_label[subset[0]:subset[1]]
        if vocabulary_set is None:
            if os.path.exists(config.VOCAB_PATH) and config.LOAD_VOCAB:
                self.vocabulary, self.idx2word = load_pickle(config.VOCAB_PATH)
            else:
                self.vocabulary, self.idx2word = get_vocabulary(self.train_data, save=True)
        else:
            self.vocabulary, self.idx2word = vocabulary_set

        self.train_data, self.train_label, augment_sentences(self.train_data, self.train_label, self.augmentations)

        self.train_data = pad_sentences(self.train_data)

        self.labels, self.idx2label = get_vocabulary(self.train_label, contains_na=False, use_wordnet=False)

        self.train_data = sentence_to_idx(self.train_data, self.vocabulary)
        self.train_label = sentence_to_idx(self.train_label, self.labels)
        self.train_label = one_hot(self.train_label, self.labels)

    def get_vocabulary_set(self):
        return self.vocabulary, self.idx2word

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
