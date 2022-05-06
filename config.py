from wordnet import *


DEVICE = 'cuda:0'
PAD = '<PAD>'
UNK = '<UNK>'
TRAIN_EPOCHS = 10
EVAL_EPOCH = 1
BATCH_SIZE = 64
LEARNING_RATE = 1e-2
DROPOUT = 0.2
MAX_SENT_LENGTH = 80

EMBEDDING_DIM = 512

LOAD_VOCAB = True
VOCAB_PATH = 'voc_set.pickle'
GENERATOR_PATH = 'gpt2-generator.pickle'
DATASET_PATH = 'ds-amazon-bt75.pickle'

wordnet = WordNet()