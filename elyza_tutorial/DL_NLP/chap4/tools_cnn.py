import numpy as np
from sklearn.utils import shuffle
from nltk import Tree

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F

random_state = 101
torch.manual_seed(101)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_DICT = {
    '0': 0,  # very negative -> negative
    '1': 0,  # negative
    '2': None,  # neutral -> 今回は使わない
    '3': 1,  # positive
    '4': 1,  # very positive -> positive
}

def load_data(file_path):
	X = []
	y = []
	with open(file_path) as f:
		for line in f:
			tree = Tree.fromstring(line)  # パース
			x = [word.lower() for word in tree.leaves()]  # ex: ['i', 'have' 'a', 'pen']
			label = LABEL_DICT[tree.label()]  # {0, 1, None}
			if label is None:  # neutralラベルは除外する
				continue
			X.append(x)
			y.append(label)
	return X, y


class DataLoaderCNN:
	def __init__(self, X, Y, batch_size, shuffle=False):
		self.data = list(zip(X, Y))
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.start_idx = 0

		self.reset()

	def reset(self):
		if self.shuffle:
			self.data = shuffle(self.data, random_state=random_state)
		self.start_idx = 0

	def __iter__(self):
		return self

	def __next__(self):
		if self.start_idx >= len(self.data):
			self.reset()
			raise StopIteration

		X, Y = zip(*self.data[self.start_idx:self.start_idx + self.batch_size])
		lengths_X = [len(sent) for sent in X]
		max_len_X = max(lengths_X)
		padded_X = [self.pad_seq(sent, max_len_X) for sent in X]

		batch_X = torch.tensor(padded_X, dtype=torch.long, device=device)
		batch_Y = torch.tensor(Y, dtype=torch.long, device=device)

		self.start_idx += self.batch_size
		return batch_X, batch_Y

	@staticmethod
	def pad_seq(seq, max_length):
		PAD = 0
		seq += [PAD for _ in range(max_length - len(seq))]
		return seq


class TextCNN(nn.Module):

	def __init__(self, vocab_size, embedding_size, class_num, kernel_num, kernel_sizes, dropout, static):
		"""
		Args:
			vocab_size : 入力言語語彙数
			embedding_size : 埋め込み次元数
			class_num : 出力のクラス数
			kernel_num : 畳み込み層の出力のチャネル数
			kernel_sizes: カーネルのウインドウサイズのリスト
			dropout : ドロップアウト率
			static : 埋め込みを固定するかのフラグ
		"""
		super().__init__()
		self.static = static
		self.vocab_size = vocab_size
		self.embedding = nn.Embedding(vocab_size, embedding_size)
		self.convs = nn.ModuleList(
			[nn.Conv1d(1, kernel_num, (kernel_size, embedding_size)) for kernel_size in kernel_sizes]
		)
		self.dropout = nn.Dropout(dropout)
		self.out = nn.Linear(len(kernel_sizes)*kernel_num, class_num)

	def forward(self, x):
		# x: (batch_size, max_length)
		x = self.embedding(x) # (batch_size, max_length, embeding_size)
		if self.static:
			x = torch.tensor(x) # 固定
		x = x.unsqueeze(1) # (batch_size, 1, max_length, embedding_size)
		x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] #[(batch_size, kernel_num, max_length-kernel_size+1)] * len(kernel_sizes)
		x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] # poolingのkernel_sizeがi.size(2)に指定されている->2次元目が1になる [(batch_size, kernel_num),...]*len(kernel_sizes)
		x = torch.cat(x, 1) # (batch_size, kernel_num*len(kernel_sizes))
		x = self.dropout(x)
		logit = self.out(x) # (batch_size, class_num)

		return logit


def compute_loss(batch_X, batch_Y, model, criterion, optimizer=None, is_train=True):
	# 損失計算
	model.train(is_train)

	if is_train:
		optimizer.zero_grad()

	pred = model(batch_X)
	loss = criterion(pred, batch_Y)

	pred_category = pred.argmax(1)
	n_correct = (pred_category == batch_Y).sum()

	if is_train:
		loss.backward()
		optimizer.step()

	return loss.item(), n_correct.item()

