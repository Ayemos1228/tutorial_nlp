import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from nltk import bleu_score
import sys
sys.path.append("..")
from tools import Vocab, ids2sentence
from tools_seq import load_data, sentence2ids, DataLoader, EncoderDecoder, compute_loss, trim_eos

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)
random_state = 42

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
BOS_TOKEN = '<S>'
EOS_TOKEN = '</S>'
PAD = 0
UNK = 1
BOS = 2
EOS = 3

MIN_COUNT = 2

word2id = {
	PAD_TOKEN: PAD,
	BOS_TOKEN: BOS,
	EOS_TOKEN: EOS,
	UNK_TOKEN: UNK,
}
hidden_size = 256
batch_size = 64
num_epochs = 10
lr = 0.01
teacher_forcing_rate =0.2
test_max_length = 20

train_X = load_data('./data/train.en')
train_Y = load_data('./data/train.ja')
test_X = load_data('./data/test.en')
test_Y = load_data('./data/test.ja')

vocab_X = Vocab(word2id=word2id)
vocab_Y = Vocab(word2id=word2id)
vocab_X.build_vocab(train_X, min_count=MIN_COUNT)
vocab_Y.build_vocab(train_Y, min_count=MIN_COUNT)
vocab_size_X = len(vocab_X.id2word)
vocab_size_Y = len(vocab_Y.id2word)

train_X, valid_X, train_Y, valid_Y = train_test_split(train_X, train_Y, test_size=0.1, random_state=random_state)
train_X = [sentence2ids(vocab_X, sent) for sent in train_X]
train_Y = [sentence2ids(vocab_Y, sent) for sent in train_Y]
valid_X = [sentence2ids(vocab_X, sent) for sent in valid_X]
valid_Y = [sentence2ids(vocab_Y, sent) for sent in valid_Y]
test_X = [sentence2ids(vocab_X, sent) for sent in test_X]
test_Y = [sentence2ids(vocab_Y, sent) for sent in test_Y]
train_dataloader = DataLoader(train_X, train_Y, batch_size)
valid_dataloader = DataLoader(valid_X, valid_Y, batch_size, shuffle=False)
test_dataloader = DataLoader(test_X, test_Y, batch_size, shuffle=False)


