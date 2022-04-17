from torch.utils.data import Dataset
import gzip
import ast

import config
from utils import *


class AmazonReviewDataset(Dataset):
    def __init__(self, filename):
        self.train_data, self.train_label = AmazonReviewDataset.parse(filename)
        max_len = max([len(x) for x in self.train_data])
        for i in range(len(self.train_data)):
            for _ in range(max_len - len(self.train_data[i])):
                self.train_data[i].append(config.PAD)

        self.vocabulary, self.idx2word = get_vocabulary(self.train_data, start_idx=1)
        self.vocabulary[config.PAD] = 0
        self.train_data = sentence_to_idx(self.train_data, self.vocabulary)
        self.labels, self.idx2label = get_vocabulary(self.train_label, start_idx=0)
        self.train_label = sentence_to_idx(self.train_label, self.labels)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item):
        return self.train_data[item], self.train_label[item][0]

    @staticmethod
    def parse(path):
        reviews = gzip.open(path, 'r')
        x = []
        y = []
        for review in reviews:
            review = review.decode('utf-8')
            review = review.replace('true', 'True')
            review = review.replace('false', 'False')
            data = ast.literal_eval(review)
            if 'reviewText' not in data:
                continue
            text = data['reviewText'].split()
            text = [word.lower() for word in text]
            x.append(text)
            y.append([data['overall']])
        return x, y