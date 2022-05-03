from wordnet import *


DEVICE = 'cuda:0'
PAD = '<PAD>'
UNK = '<UNK>'
TRAIN_EPOCHS = 100
EVAL_EPOCH = 5
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
DROPOUT = 0.2

EMBEDDING_DIM = 300

LOAD_VOCAB = True
VOCAB_PATH = 'voc_set.pickle'

wordnet = WordNet()