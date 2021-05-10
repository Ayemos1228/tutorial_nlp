import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tools import Vocab, load_data, sentence2ids, pad_seq

# device設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD = 0

class DataLoaderSG:
	def __init__(self, text, batch_size, window=3):
		self.text = text
		self.batch_size = batch_size
		self.window = window
		self.s_pointer = 0
		self.w_pointer = 0

	def __iter__(self):
		return self

	def __next__(self):
		batch_X = []
		batch_Y = []

		while len(batch_X) < self.batch_size:
			sentence = self.text[self.s_pointer]
			# 予測元
			context = sentence[self.w_pointer]

			# 予測先
			begin = max(0, self.w_pointer - self.window)
			# end = min(len(sentence), self.w_pointer + self.window + 1)
			# target = sentence[begin:self.w_pointer] + sentence[self.w_pointer + 1:end]
			target = sentence[begin:self.w_pointer] + sentence[self.w_pointer + 1:self.w_pointer + self.window + 1]
			target = pad_seq(target, self.window * 2)

			batch_X.append(context)
			batch_Y.append(target)

			# ポインター更新
			self.w_pointer += 1
			if self.w_pointer >= len(sentence):
				self.w_pointer = 0
				self.s_pointer += 1
				if self.s_pointer >= len(self.text):
					self.s_pointer = 0
					self.w_pointer = 0
					raise StopIteration

		batch_X = torch.tensor(batch_X, dtype=torch.long, device=device)
		batch_Y = torch.tensor(batch_Y, dtype=torch.long, device=device)
		return batch_X, batch_Y


class SkipGram(nn.Module):
	def __init__(self, vocab_size, embedding_size):
		super().__init__()
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
		self.linear = nn.Linear(self.embedding_size, self.vocab_size, bias=False)

	def forward(self, batch_X, batch_Y):
		"""
		Args:
			batch_X (tensor): (batch_size, )
			batch_Y (tensor): (batch_size, window * 2)
		Returns:
			loss (torch.Tensor(dtype=torch.float)): loss
		"""
		emb_X = self.embedding(batch_X) # (batch_size, embedding_size)
		lin_X = self.linear(emb_X) # (batch_size, vocab_size)
		log_prob_X = F.log_softmax(lin_X, dim=-1) # (batch_size, vocab_size)
		log_prob_X = torch.gather(log_prob_X, 1, batch_Y) # (batch_size, window * 2)
		log_prob_X = log_prob_X * (batch_Y != PAD).float()
		loss = log_prob_X.sum(1).mean().neg()

		return loss



# negative samplingに使う確率分布
# weights = np.power([0, 0] + list(vocab.raw_vocab.values()), 0.75)
# weights = weights / weights.sum()

class DataLoaderSGNS:
	# negative sampling付きのskipgram
	def __init__(self, text, batch_size, window=3, n_negative=5, weights=None):
		self.text = text
		self.batch_size = batch_size
		self.window = window
		self.n_negative = n_negative
		self.weights = None
		if weights is not None:
			self.weights = torch.FloatTensor(weights)
		self.s_pointer = 0
		self.w_pointer = 0

	def __iter__(self):
		return self

	def __next__(self):
		batch_X = []
		batch_Y = []
		batch_N = []

		while len(batch_X) < self.batch_size:
			sentence = self.text[self.s_pointer]
			context = sentence[self.w_pointer]

			begin = max(0, self.w_pointer - self.window)
			end = min(len(sentence), self.w_pointer + self.window + 1)
			target = sentence[begin:self.w_pointer] + sentence[self.w_pointer + 1:end]
			target = pad_seq(target, self.window * 2)
			batch_X.append(context)
			batch_Y.append(target)

			# 負例を追加
			# 負例のうちに正例が含まれないよう修正
			# tmp_weights = torch.clone(self.weights)
			# for positive_id in target:
			# 	tmp_weights[positive_id] = 0
			# negative_samples = torch.multinomial(tmp_weights, self.n_negative) # (n_negative,)
			# 負例に正例が含まれる可能性がある
			negative_samples = torch.multinomial(self.weights, self.n_negative) # (n_negative,)
			batch_N.append(negative_samples.unsqueeze(0)) # (1, n_negative)
			self.w_pointer += 1
			if self.w_pointer >= len(sentence):
				self.w_pointer = 0
				self.s_pointer += 1
				if self.s_pointer >= len(self.text):
					self.s_pointer = 0
					self.w_pointer = 0
					raise StopIteration

		batch_X = torch.tensor(batch_X, dtype=torch.long, device=device)
		batch_Y = torch.tensor(batch_Y, dtype=torch.long, device=device)
		batch_N = torch.cat(batch_N, dim=0).to(device) # (batch_size, n_negative)

		return batch_X, batch_Y, batch_N

class SkipGramNS(nn.Module):
	def __init__(self, vocab_size, embedding_size):
		super(SkipGramNS,self).__init__()
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.in_embedding = nn.Embedding(vocab_size, embedding_size)
		self.out_embedding = nn.Embedding(vocab_size, embedding_size)

	def forward(self, batch_X, batch_Y, batch_N):
		"""
		Args:
			batch_X : (batch_size, )
			batch_Y : (batch_size, window * 2)
			batch_N : (batch_size, n_negative)
		"""
		emb_X = self.in_embedding(batch_X).unsqueeze(2) # (batch_size, embedding_size, 1)
		emb_Y = self.out_embedding(batch_Y) # (batch_size, window * 2, embedding_size)
		emb_N = self.out_embedding(batch_N) # (batch_size, n_negative, embedding_size)
		loss_Y = torch.bmm(emb_Y, emb_X).squeeze().sigmoid().log() # (batch_size, window * 2)
		loss_Y = loss_Y * (batch_Y != PAD).float()
		loss_Y = loss_Y.sum(1) # (batch_size, )
		loss_N = torch.bmm(emb_N, emb_X).squeeze().neg().sigmoid().log().sum(1) # (batch_size, )
		return -(loss_Y + loss_N).mean()


