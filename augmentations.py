import random

from torch.utils.data import Dataset

from utils import *
import config


class Augmentation:
    def __init__(self):
        pass

    def forward(self, x):
        # x: list of words
        pass

    def __call__(self, x):
        return self.forward(x)


class Compose(Augmentation):
    def __init__(self, augmentations):
        super(Compose, self).__init__()
        self.augmentations = augmentations

    def forward(self, x):
        for augmentation in self.augmentations:
            x = augmentation(x)
        return x


class EasyAug(Augmentation):
    def __init__(self, n=1, p=0.1, sr=True, ri=True, rs=True, rd=True):
        super(EasyAug, self).__init__()
        self.n = n
        self.p = p
        self.use_sr, self.use_ri, self.use_rs, self.use_rd = sr, ri, rs, rd

    def forward(self, x):
        if self.use_sr:
            # synonym replacement
            x = self.sr(x)
        if self.use_ri:
            # random insertions
            x = self.ri(x)
        if self.use_rs:
            # random swap
            x = self.rs(x)
        if self.use_rd:
            # random deletion
            x = self.rd(x)
        return x

    def sr(self, x):
        idxs = self.sample_indices(x)
        for idx in idxs:
            x[idx] = config.wordnet.get_random_synonym(x[idx])
        return x

    def ri(self, x):
        idxs = self.sample_indices(x)
        syns = [config.wordnet.get_random_synonym(x[i]) for i in idxs]
        for i in range(self.n):
            r = random.randint(0, len(x))
            x.insert(r, syns[min(i, len(syns) - 1)])
        return x

    def rs(self, x):
        for i in range(self.n):
            p1, p2 = random.sample(range(len(x)), 2)
            x[p1], x[p2] = x[p2], x[p1]
        return x

    def rd(self, x):
        for i in range(len(x) - 1, -1, -1):
            if random.random() < self.p:
                x.pop(i)
        return x

    def sample_indices(self, x):
        idxs = not_stopword_indices(x)
        return random.sample(idxs, k=self.n)


def augment_sentences(sentences, labels, augmentation, n=-1):
    if isinstance(augmentation, list):
        if len(augmentation) == 0:
            return
        augmentation = Compose(augmentation)
    if n == -1:
        n = len(sentences)
    idxs = random.sample(range(len(sentences)), n)
    for i in range(n):
        sentences.append(augmentation(sentences[idxs[i]]))
        labels.append(labels[idxs[i]])
