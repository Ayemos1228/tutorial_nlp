import numpy as np
from sklearn.utils import shuffle
from nltk import Tree

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import sys
from tools_cnn import load_data, DataLoaderCNN, TextCNN, compute_loss
sys.path.append("..")
from tools import Vocab, sentence2ids, pad_seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)
random_state = 42

PAD = 0
UNK = 1

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'

LABEL_DICT = {
	'0': 0,  # very negative -> negative
	'1': 0,  # negative
	'2': None,  # neutral -> 今回は使わない
	'3': 1,  # positive
	'4': 1,  # very positive -> positive
}

word2id = {
	PAD_TOKEN : PAD,
	UNK_TOKEN : UNK,
}

# データの読み込み
train_X, train_Y = load_data('./trees/train.txt')
valid_X, valid_Y = load_data('./trees/dev.txt')
train_X = train_X[:len(train_X)//2]
train_Y = train_Y[:len(train_Y)//2]
valid_X = valid_X[:len(valid_X)//2]
valid_Y = valid_Y[:len(valid_Y)//2]

vocab = Vocab(word2id=word2id)
vocab.build_vocab(train_X)

train_X = [sentence2ids(vocab, x) for x in train_X]
valid_X = [sentence2ids(vocab, x) for x in valid_X]

# parameters
lr = 0.001
num_epochs = 30
batch_size = 128
ckpt_path = "cnn.pth"
vocab_size = len(vocab.word2id)
embedding_size = 128
class_num = 2
kernel_num = 64
kernel_sizes = [3, 4, 5]
dropout = 0.5
static = False

model = TextCNN(vocab_size, embedding_size, class_num, kernel_num, kernel_sizes, dropout, static).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

train_dataloader = DataLoaderCNN(train_X, train_Y, batch_size, shuffle=True)
valid_dataloader = DataLoaderCNN(valid_X, valid_Y, batch_size, shuffle=False)



# 学習
best_valid_acc = 0
for epoch in range(1, num_epochs+1):
	train_losses = []
	train_corrects = 0
	valid_losses = []
	valid_corrects = 0
	# train
	for batch in train_dataloader:
		batch_X, batch_Y = batch
		train_loss, train_correct = compute_loss(
			batch_X, batch_Y, model, criterion, optimizer, is_train=True
			)
		train_losses.append(train_loss)
		train_corrects += train_correct
	# valid
	for batch in valid_dataloader:
		batch_X, batch_Y = batch
		valid_loss, valid_correct = compute_loss(
			batch_X, batch_Y, model, criterion, is_train=False
			)
		valid_losses.append(valid_loss)
		valid_corrects += valid_correct
	train_loss = np.mean(train_losses)
	valid_loss = np.mean(valid_losses)
	train_acc = 100. * train_corrects / len(train_dataloader.data)
	valid_acc = 100. * valid_corrects / len(valid_dataloader.data)

	if valid_acc > best_valid_acc:
		ckpt = model.state_dict()
		torch.save(ckpt, ckpt_path)
		best_valid_acc = valid_acc

	print('Epoch {}: train_loss: {:.4f}  train_acc: {:.2f}  valid_loss: {:.4f}  valid_acc: {:.2f}'.format(
			epoch, train_loss, train_acc, valid_loss, valid_acc))
	print('-'*80)
