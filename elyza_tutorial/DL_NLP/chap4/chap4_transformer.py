import numpy as np
import time
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import bleu_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
from tools_transformer import load_data, sentence2ids, Transformer, DataLoader, compute_loss, calc_bleu
sys.path.append("..")
from tools import Vocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(101)
random_state = 101
PAD = 0
UNK = 1
BOS = 2
EOS = 3
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
BOS_TOKEN = '<S>'
EOS_TOKEN = '</S>'
word2id = {
	PAD_TOKEN: PAD,
	BOS_TOKEN: BOS,
	EOS_TOKEN: EOS,
	UNK_TOKEN: UNK,
}
MIN_COUNT = 2

train_X = load_data('./data/train.en')
train_Y = load_data('./data/train.ja')
# 演習用にデータサイズを縮小
train_X = train_X[:len(train_X)//2]
train_Y = train_Y[:len(train_Y)//2]
train_X, valid_X, train_Y, valid_Y = train_test_split(train_X, train_Y, test_size=0.2, random_state=random_state)



vocab_X = Vocab(word2id=word2id)
vocab_Y = Vocab(word2id=word2id)
vocab_X.build_vocab(train_X, min_count=MIN_COUNT)
vocab_Y.build_vocab(train_Y, min_count=MIN_COUNT)

vocab_size_X = len(vocab_X.id2word)
vocab_size_Y = len(vocab_Y.id2word)
train_X = [sentence2ids(vocab_X, sentence) for sentence in train_X]
train_Y = [sentence2ids(vocab_Y, sentence) for sentence in train_Y]
valid_X = [sentence2ids(vocab_X, sentence) for sentence in valid_X]
valid_Y = [sentence2ids(vocab_Y, sentence) for sentence in valid_Y]

MAX_LENGTH = 20
batch_size = 64
num_epochs = 15
lr = 0.001
ckpt_path = 'transformer.pth'
max_length = MAX_LENGTH + 2

model_args = {
	'input_vocab': vocab_size_X,
	'output_vocab': vocab_size_Y,
	'max_length': max_length,
	'proj_share_weight': True,
	'd_k': 32,
	'd_v': 32,
	'd_model': 128,
	'embedding_size': 128,
	'd_inner_hid': 256,
	'n_layers': 3,
	'n_head': 6,
	'dropout': 0.1,
}

train_dataloader = DataLoader(train_X, train_Y, batch_size)
valid_dataloader = DataLoader(valid_X, valid_Y, batch_size,shuffle=False)

model = Transformer(**model_args).to(device)
optimizer = optim.Adam(model.get_trainable_parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(ignore_index=PAD, size_average=False).to(device)

best_valid_bleu = 0.

for epoch in range(1, num_epochs+1):
	start = time.time()
	train_loss = 0.
	train_refs = []
	train_hyps = []
	valid_loss = 0.
	valid_refs = []
	valid_hyps = []
	# train
	for batch in train_dataloader:
		batch_X, batch_Y = batch
		loss, gold, pred = compute_loss(
			batch_X, batch_Y, model, criterion, optimizer, is_train=True
			)
		train_loss += loss
		train_refs += gold
		train_hyps += pred
	# valid
	for batch in valid_dataloader:
		batch_X, batch_Y = batch
		loss, gold, pred = compute_loss(
			batch_X, batch_Y, model, criterion, is_train=False
			)
		valid_loss += loss
		valid_refs += gold
		valid_hyps += pred
	# 損失をサンプル数で割って正規化
	train_loss /= len(train_dataloader.data)
	valid_loss /= len(valid_dataloader.data)
	# BLEUを計算
	train_bleu = calc_bleu(train_refs, train_hyps)
	valid_bleu = calc_bleu(valid_refs, valid_hyps)

	# validationデータでBLEUが改善した場合にはモデルを保存
	if valid_bleu > best_valid_bleu:
		ckpt = model.state_dict()
		torch.save(ckpt, ckpt_path)
		best_valid_bleu = valid_bleu

	elapsed_time = (time.time()-start) / 60
	print('Epoch {} [{:.1f}min]: train_loss: {:5.2f}  train_bleu: {:2.2f}  valid_loss: {:5.2f}  valid_bleu: {:2.2f}'.format(
			epoch, elapsed_time, train_loss, train_bleu, valid_loss, valid_bleu))
	print('-'*80)

