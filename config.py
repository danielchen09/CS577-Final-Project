from wordnet import *


DEVICE = 'cuda:0'
PAD = '<PAD>'
UNK = '<UNK>'
TRAIN_EPOCHS = 100
EVAL_EPOCH = 1
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
DROPOUT = 0.2
MAX_SENT_LENGTH = 80

EMBEDDING_DIM = 300

LOAD_VOCAB = True
VOCAB_PATH = 'voc_set.pickle'

wordnet = WordNet()