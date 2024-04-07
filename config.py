ORIGIN_DIR = './inputs/origin/'
ANNOTATION_DIR = './outputs/annotation/'

TRAIN_SAMPLE_PATH = './outputs/train_sample.txt'
VAL_SAMPLE_PATH = './outputs/validation_sample.txt'
TEST_SAMPLE_PATH = './outputs/test_sample.txt'

VOCAB_PATH = './outputs/vocab.txt'
LABEL_PATH = './outputs/label.txt'

WORD_PAD = '<PAD>'
WORD_UNK = '<UNK>'

WORD_PAD_ID = 0
WORD_UNK_ID = 1
LABEL_O_ID = 0

VOCAB_SIZE = 1647
EMBEDDING_DIM = 100
HIDDEN_SIZE = 256
TARGET_SIZE = 27
LR = 3e-5
EPOCH = 100

MODEL_DIR = './outputs/model/'

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# bert改造
BERT_MODEL = './huggingface/chinese-macbert-base'
EMBEDDING_DIM = 768
MAX_POSITION_EMBEDDINGS = 512