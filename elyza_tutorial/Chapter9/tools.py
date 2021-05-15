import random
import re
import pickle
import gensim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

# vocab構築のハイパーパラメター
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
PAD = 0
UNK = 1
MIN_COUNT = 1

word2id = {
PAD_TOKEN: PAD,
UNK_TOKEN: UNK,
}

random_state = 42
torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(file_path):
	titles = []
	labels = []
	with open(file_path, "r") as f:
		for line in f:
			if "TITLE" in line:
				continue
			line_split = line.strip().split("\t")
			sent = line_split[0].lower()
			sent = re.sub(r'\d+', '0', sent)
			sent = re.sub(r'[^ \w]', '', sent)
			words = sent.split()
			titles.append(words)
			labels.append(line_split[1])

	return titles, labels


class Vocab:
	def __init__(self, word2id={}):
		self.word2id = dict(word2id)
		self.id2word = {v: k for k, v in self.word2id.items()}

	def build_vocab(self, corpus, min_count=1):
		"""コーパスから辞書を構築するメソッド

		Args:
			corpus (list of list of str): コーパス
			min_count (int, optional): 辞書に含める最小単語出現回数 Defaults to 1.
		"""
		word_counter = {}
		for sentence in corpus:
			for word in sentence:
				word_counter[word] = word_counter.get(word, 0) + 1

		# 最小出現回数を考慮
		for word, count in sorted(word_counter.items(), key=lambda x: -x[1]):
			if count < min_count:
				break
			_id = len(self.word2id)
			self.word2id.setdefault(word, _id)
			self.id2word[_id] = word

		# option
		self.raw_vocab = {w: word_counter[w] for w in self.word2id.keys() if w in word_counter}


def sentence2ids(vocab, sentence):
	"""単語のリストとしての文を、idのリストに変換する関数

	Args:
		vocab (vocab classのobject): 語彙
		sentence (list of str): 単語のリスト（mecabでの処理ずみ)
	Returns:
		ids (list of int): 単語IDのリスト
	"""
	ids = [vocab.word2id.get(word, UNK) for word in sentence]
	return ids


def pad_seq(seq, max_length):
	"""paddingを行う関数。

	Args:
		seq (list of int): 単語idのリスト（文に対応）
		max_length (int): バッチ内の文の最大長
	Returns:
		seq (list of int): padding後のリスト
	"""
	seq += [PAD for i in range(max_length - len(seq))]
	return seq


def get_embedding(vocab, embedding_path):
	"""学習済みembeddingからembeddingを取得する関数。
	もしも、vocab.word2id内のエントリがなかった場合はランダムに初期化
	Args:
		vocab : vocabクラスのオブジェクト
		embedding_path : 学習済みembeddingのpath
	Returns:
		embedding: 単語埋め込みのweights (vocab_size, 300)
	"""
	with open(embedding_path, "rb") as f:
		vec = pickle.load(f)
	embedding = []
	for word in vocab.word2id.keys():
		if word in vec.key_to_index.keys():
			embedding.append(vec[word])
		else:
			embedding.append(np.random.rand(300))
	embedding = torch.tensor(embedding, dtype=torch.float, device=device)
	return embedding


def compute_loss(batch_X, batch_Y, lengths_X, model, optimizer=None, is_train=True):

	model.train(is_train)
	logits = model(batch_X, lengths_X)
	loss = F.cross_entropy(logits, batch_Y).to(device)

	if is_train:
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	pred = logits.argmax(-1).view(batch_Y.size())
	n_corrects = (batch_Y.data == pred.data).sum()
	return loss.item(), n_corrects

def compute_loss_cnn(batch_X, batch_Y, model, optimizer=None, is_train=True):

	model.train(is_train)
	logits = model(batch_X)
	loss = F.cross_entropy(logits, batch_Y).to(device)

	if is_train:
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	pred = logits.argmax(-1).view(batch_Y.size())
	n_corrects = (batch_Y.data == pred.data).sum()
	return loss.item(), n_corrects

# RNN
class DataLoaderRNN:
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

		batch_X, batch_Y = zip(*self.data[self.start_idx:self.start_idx + self.batch_size])
		batch_pairs = sorted(zip(batch_X, batch_Y), key=lambda x: len(x[0]), reverse=True)
		batch_X, batch_Y = zip(*batch_pairs)
		lengths_X = [len(sent) for sent in batch_X]
		max_len_X = max(lengths_X)
		padded_X = [pad_seq(sent, max_len_X) for sent in batch_X]

		batch_X = torch.tensor(padded_X, dtype=torch.long, device=device).transpose(0, 1)
		batch_Y = torch.tensor(batch_Y, dtype=torch.long, device=device)

		self.start_idx += self.batch_size
		return batch_X, batch_Y, lengths_X

class RNN(nn.Module):
	def __init__(self, input_size, embedding_size, hidden_size, num_layers=1, embedding=None):
		"""
		Args:
			input_size : 入力言語の語彙数
			hidden_size : 隠れ層のユニット数
		"""
		super().__init__()
		if embedding is not None:
			self.embedding = nn.Embedding.from_pretrained(embedding, padding_idx=PAD)
			self.embedding_size = 300
		else:
			self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=PAD)
			self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.rnn = nn.RNN(embedding_size, hidden_size, num_layers)
		self.linear = nn.Linear(hidden_size, 4)

	def forward(self, batch_X, input_lengths, hidden=None):
		"""
		Args:
			batch: paddingされた入力バッチ系列 (max_length, batch_size)
			input_lengths: 入力の各バッチごとの長さのリスト
			hidden ([type], optional): 隠れ状態. Defaults to None.
		Returns:
			logits: モデルの出力　(batch_size, 4)
		"""
		emb = self.embedding(batch_X) # (max_length, batch_size, hidden_size)
		packed = pack_padded_sequence(emb, input_lengths)
		output, hidden = self.rnn(packed, hidden)
		output, _ = pad_packed_sequence(output)
		output_n = output[-1] # (batch_size, hidden_size)
		logits = self.linear(output_n)
		return logits


class BiRNN(nn.Module):
	def __init__(self, input_size, embedding_size, hidden_size, num_layers=1, embedding=None):
		"""
		Args:
			input_size : 入力言語の語彙数
			hidden_size : 隠れ層のユニット数
		"""
		super().__init__()
		if embedding is not None:
			self.embedding = nn.Embedding.from_pretrained(embedding, padding_idx=PAD)
			self.embedding_size = 300
		else:
			self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=PAD)
			self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, bidirectional=True)
		self.linear = nn.Linear(2 * hidden_size, 4)

	def forward(self, batch_X, input_lengths, hidden=None):
		"""
		Args:
			batch: paddingされた入力バッチ系列 (max_length, batch_size)
			input_lengths: 入力の各バッチごとの長さのリスト
			hidden ([type], optional): 隠れ状態. Defaults to None.
		Returns:
			logits: モデルの出力　(batch_size, 4)
		"""
		emb = self.embedding(batch_X) # (max_length, batch_size, hidden_size)
		packed = pack_padded_sequence(emb, input_lengths)
		output, _ = self.rnn(packed, hidden) # (max_length, batch_size, 2 * hidden_size)
		output, _ = pad_packed_sequence(output)
		output_n = output[-1] # (batch_size, 2 * hidden_size)
		logits = self.linear(output_n)
		return logits

# CNN
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

		batch_X, batch_Y = zip(*self.data[self.start_idx:self.start_idx + self.batch_size])
		lengths_X = [len(sent) for sent in batch_X]
		max_len_X = max(lengths_X)
		padded_X = [pad_seq(sent, max_len_X) for sent in batch_X]

		batch_X = torch.tensor(padded_X, dtype=torch.long, device=device) #(batch_size, max_len_X)
		batch_Y = torch.tensor(batch_Y, dtype=torch.long, device=device) # (batch_size, )

		self.start_idx += self.batch_size
		return batch_X, batch_Y


class CNN(nn.Module):

	def __init__(self, input_size, embedding_size, kernel_num, kernel_size, embedding=None):
		super().__init__()
		self.input_size = input_size
		if embedding is not None:
			self.embedding = nn.Embedding.from_pretrained(embedding, padding_idx=PAD)
			self.embedding_size = 300
		else:
			self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=PAD)
			self.embedding_size = embedding_size
		self.conv = nn.Conv1d(self.embedding_size, kernel_num, kernel_size, padding=1)
		self.linear = nn.Linear(kernel_num, 4)

	def forward(self, X):
		# X: (batch_size, max_len)
		emb_X = self.embedding(X) # (batch_size, max_len, embedding_size)
		emb_X = emb_X.transpose(1, 2) # (batch_size, embedding_size, max_len)
		conv_X = self.conv(emb_X) # (batch_size, kernel_num, maxlen(-kernel_size+1))
		pool_X = F.max_pool1d(conv_X, conv_X.size(2)).squeeze(2) # (batch_size, kernel_num)
		logits = self.linear(pool_X) # (batch_size, 4)
		return logits


class TextCNN(nn.Module):

	def __init__(self, vocab_size, embedding_size, class_num, kernel_num, kernel_sizes, dropout, static, embedding=None):
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
		if embedding is not None:
			self.embedding = nn.Embedding.from_pretrained(embedding, padding_idx=PAD)
			self.embedding_size = 300
		else:
			self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=PAD)
			self.embedding_size = embedding_size
		self.convs = nn.ModuleList(
			[nn.Conv1d(1, kernel_num, (kernel_size, self.embedding_size)) for kernel_size in kernel_sizes]
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
