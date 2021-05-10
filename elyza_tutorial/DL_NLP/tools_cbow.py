import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tools import pad_seq

# device設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD = 0

class DataLoaderCBOW:
	def __init__(self, text, batch_size, window=3):
		"""
		Args:
			text (list of list of int):  単語IDに変換したデータセット
			batch_size (int): ミニバッチのサイズ
			window (int, optional): ウィンドウサイズ Defaults to 3.
		"""
		self.text = text
		self.batch_size = batch_size
		self.window = window
		self.s_pointer = 0 # 文を走査するポインター
		self.w_pointer = 0 # 単語を走査するポインター

	def __iter__(self):
		return self

	def __next__(self):
		batch_X = []
		batch_Y = []

		while len(batch_X) < self.batch_size:
			sentence = self.text[self.s_pointer]
			w_target = sentence[self.w_pointer]

			# 文脈ベクトルを取得
			begin = max(0, self.w_pointer - self.window)
			end = min(len(sentence), self.w_pointer + self.window + 1)
			context = sentence[begin:self.w_pointer] + sentence[self.w_pointer + 1:end]
			context = pad_seq(context, self.window * 2)

			# バッチに入れる
			batch_X.append(context)
			batch_Y.append(w_target)

			# ポインターの更新
			self.w_pointer += 1
			if self.w_pointer >= len(sentence):
				self.w_pointer = 0
				self.s_pointer += 1

			if self.s_pointer >= len(self.text):
				self.s_pointer = 0
				raise StopIteration

		batch_X = torch.tensor(batch_X, dtype=torch.long, device=device)
		batch_Y = torch.tensor(batch_Y, dtype=torch.long, device=device)
		return batch_X, batch_Y



class CBOW(nn.Module):
	def __init__(self, vocab_size, embedding_size):
		super().__init__()

		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
		self.linear = nn.Linear(self.embedding_size, self.vocab_size, bias=False)

	def forward(self, batch_X, batch_Y):
		"""
		Args:
			batch_X (torch.Tensor(dtype=torch.long)): (batch_size, window*2)
			batch_Y	(torch.Tensor(dtype=torch.long)): (batch_size,)
		Returns:
			loss (torch.Tensor(dtype=torch.float)) : CBOWの損失
		"""

		emb_X = self.embedding(batch_X) # (batch_size, window*2, embedding_size)
		emb_X = emb_X * (batch_X != PAD).float().unsqueeze(-1) # (batch_size, window*2, embedding_size)
		sum_X = torch.sum(emb_X, dim=1) # (batch_size, embedding_size) 予測にはcontext単語ベクトルの和を使う
		lin_X = self.linear(sum_X)
		loss = F.nll_loss(F.log_softmax(lin_X, dim=-1), batch_Y)
		return loss

