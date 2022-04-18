from wordnet import *


DEVICE = 'cuda:0'
PAD = '<PAD>'
TRAIN_EPOCHS = 100
EVAL_EPOCH = 5
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
DROPOUT = 0.2

EMBEDDING_DIM = 300

wordnet = WordNet()