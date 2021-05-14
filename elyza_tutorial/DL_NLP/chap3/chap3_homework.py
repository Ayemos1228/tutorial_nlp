import csv
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
from tools_seq import load_data, sentence2ids, DataLoader, EncoderDecoder, compute_loss, trim_eos, calc_bleu

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


model = EncoderDecoder(vocab_size_X, vocab_size_Y, hidden_size)
optimizer = optim.Adam(model.parameters(), lr=lr)

best_valid_bleu = 0.
for epoch in range(1, num_epochs + 1):
	train_loss = 0.
	train_refs = []
	train_hyps = []
	valid_loss = 0.
	valid_refs = []
	valid_hyps = []
	for batch in train_dataloader:
		batch_X, batch_Y, lengths_X = batch
		loss, gold, pred = compute_loss(batch_X, batch_Y, lengths_X, model, optimizer, is_train=True, teacher_forcing_rate=teacher_forcing_rate)
		train_loss += loss
		train_refs += gold
		train_hyps += pred

	for batch in valid_dataloader:
		batch_X, batch_Y, lengths_X = batch
		loss, gold, pred = compute_loss(batch_X, batch_Y, lengths_X, model, optimizer, is_train=False)
		valid_loss += loss
		valid_refs += gold
		valid_hyps += pred

	train_loss /= len(train_dataloader.data)
	valid_loss /= len(valid_dataloader.data)

	train_bleu = calc_bleu(train_refs, train_hyps)
	valid_bleu = calc_bleu(valid_refs, valid_hyps)

	if valid_bleu > best_valid_bleu:
		ckpt = model.state_dict()
		best_valid_bleu = valid_bleu

	print('Epoch {}: train_loss: {:5.2f}  train_bleu: {:2.2f}  valid_loss: {:5.2f}  valid_bleu: {:2.2f}'.format(
			epoch, train_loss, train_bleu, valid_loss, valid_bleu))
	print('-'*80)


model.load_state_dict(ckpt)
model.eval()

test_loss = 0.
test_refs = []
test_hyps = []
for batch in test_dataloader:
	batch_X, batch_Y, lengths_X = batch
	loss, gold, pred = compute_loss(batch_X, batch_Y, lengths_X, model, is_train=False)
	test_loss += loss
	test_refs += gold
	test_hyps += pred

test_loss /= len(test_dataloader.data)
test_bleu = calc_bleu(test_refs, test_hyps)

print("Test loss: {:5.2f} test_bleu: {:2.2f}".format(test_loss, test_bleu))

with open('submission.csv', 'w') as f:
    writer = csv.writer(f, delimiter=' ', lineterminator='\n')
    writer.writerows(test_hyps)
