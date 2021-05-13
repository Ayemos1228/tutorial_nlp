import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from nltk import bleu_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
BOS_TOKEN = '<S>'
EOS_TOKEN = '</S>'
PAD = 0
UNK = 1
BOS = 2
EOS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)
random_state = 42

def load_data(file_path):
	"""
		ファイルを読み込む関数
	"""
	data = []
	for line in open(file_path, "r"):
		words = line.strip().split()
		data.append(words)

	return data

def sentence2ids(vocab, sentence):
	"""単語のリストとしての文を、idのリストに変換する関数（EOSつき）

	Args:
		vocab (vocab classのobject): 語彙
		sentence (list of str): 単語のリスト（mecabでの処理ずみ)
	Returns:
		ids (list of int): 単語IDのリスト
	"""
	ids = [vocab.word2id.get(word, UNK) for word in sentence]
	ids.append(EOS)
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

class DataLoader:
	def __init__(self, X, Y, batch_size, shuffle=False):
		"""
		Args:
			X : 入力言語の文章のリスト
			Y : 出力言語の文章のリスト
			batch_size : バッチサイズ
			shuffle (bool, optional): サンプルをシャッフルするか Defaults to False.
		"""
		self.data = list(zip(X, Y))
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.start_index = 0

		self.reset()

	def reset(self):
		if self.shuffle:
			self.data = shuffle(self.data, random_state=random_state)
		self.start_index = 0

	def __iter__(self):
		return self

	def __next__(self):
		if self.start_index >= len(self.data):
			self.reset()
			raise StopIteration

		batch_X, batch_Y = zip(*self.data[self.start_index:self.start_index + self.batch_size])
		# batchを長さで降順にソートする
		batch_pairs = sorted(zip(batch_X, batch_Y), key=lambda x: len(x[0]), reverse=True)
		batch_X, batch_Y = zip(*batch_pairs)
		# パディング
		lengths_X = [len(s) for s in batch_X]
		lengthx_Y = [len(s) for s in batch_Y]
		max_len_X = max(lengths_X)
		max_len_Y = max(lengthx_Y)
		padded_X = [pad_seq(s, max_len_X) for s in batch_X] # (batch_size, max_len_X)
		padded_Y = [pad_seq(s, max_len_Y) for s in batch_Y] # (batch_size, max_len_Y)
		# バッチをまたいで各時刻ごとに学習するので転置する
		batch_X = torch.tensor(padded_X, dtype=torch.long, device=device).transpose(0, 1) # (max_len_X, batch_size)
		batch_Y = torch.tensor(padded_Y, dtype=torch.long, device=device).transpose(0, 1) # (max_len_Y, batch_size)

		self.start_index += self.batch_size
		return batch_X, batch_Y, lengths_X


class Encoder(nn.Module):
	def __init__(self, input_size, hidden_size):
		"""
		Args:
			input_size : 入力言語の語彙数
			hidden_size : 隠れ層のユニット数
		"""
		super().__init__()
		self.hidden_size = hidden_size

		self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=PAD)
		self.gru = nn.GRU(hidden_size, hidden_size)

	def forward(self, batch_X, input_lengths, hidden=None):
		"""
		Args:
			batch: paddingされた入力バッチ系列 (max_length, batch_size)
			input_lengths: 入力の各バッチごとの長さのリスト
			hidden ([type], optional): 隠れ状態. Defaults to None.
		Returns:
			output: Encoderの出力 (max_length, batch_size, hidden_size)
			hidden: Encoderの隠れ状態 (1, batch_size, hidden_size)
		"""
		emb = self.embedding(batch_X) # (max_length, batch_size, hidden_size)
		packed = pack_padded_sequence(emb, input_lengths)
		output, hidden = self.gru(packed, hidden)
		output, _ = pad_packed_sequence(output)
		return output, hidden


class Decoder(nn.Module):
	def __init__(self, hidden_size, output_size):
		"""
		Args:
			hidden_size: 隠れ層のunit数
			output_size: 出力出力言語の語彙数
		"""
		super().__init__()
		self.hidden_size = hidden_size
		self.output_size = output_size

		self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=PAD)
		self.gru = nn.GRU(hidden_size, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)

	def forward(self, seqs, hidden):
		"""
		Args:
			seqs : 入力のバッチ (1, batch_size)
			hidden : 隠れ状態の初期値
		Returns:
			output: Decoderの出力 (1, batch_size, output_size)
			hidden: Decoderの隠れ状態 (1, batch_size, hidden_size)
		"""
		emb = self.embedding(seqs) # (1, batch_size, hidden_size)
		output, hidden = self.gru(emb, hidden)
		output = self.out(output)
		return output, hidden


class EncoderDecoder(nn.Module):
	def __init__(self, input_size, output_size, hidden_size):
		"""
		Args:
			input_size :  入力言語語彙数
			output_size : 入力言語語彙数
			hidden_size : 隠れ層ユニット数
		"""
		super().__init__()
		self.encoder = Encoder(input_size, hidden_size)
		self.decoder = Decoder(hidden_size, output_size)

	def forward(self, batch_X, lengths_X, max_length, batch_Y=None, use_teacher_forcing=False):
		"""
		Args:
			batch_X : 入力系列のバッチ (max_length, batch_size)
			lengths_X : 入力のバッチごとの長さのリスト
			max_length : Docoderの最大文長
			batch_Y ([type], optional): Decoderのターゲット系列. Defaults to None.
			use_teacher_forcing (bool, optional): teacherforcingをするかどうか. Defaults to False.
		Returns:
			decoder_outputs: 出力 (max_length, batch_size, seff.decoder.output_size)
		"""
		_, encoder_hidden = self.encoder(batch_X, lengths_X)
		_batch_size = batch_X.size(1)

		decoder_input = torch.tensor([BOS] * _batch_size, dtype=torch.long, device=device) # (batch_size, )
		decoder_input = decoder_input.unsqueeze(0) # (1, batch_size)
		decoder_hidden = encoder_hidden # encoderの最終隠れ状態

		decoder_outputs = torch.zeros(max_length, _batch_size, self.decoder.output_size, device=device)

		for t in range(max_length):
			decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
			decoder_outputs[t] = decoder_output

			if use_teacher_forcing and batch_Y is not None:
				decoder_input = batch_Y[t].unsqueeze(0) # (1, batch_size)
			else:
				decoder_input = decoder_output.argmax(-1) # (1, batch_size)
				# decoder_input = decoder_output.max(-1)[1]も同じ

		return decoder_outputs


def masked_cross_entropy(logits, target):
	"""
	Args:
		logits : decoderの出力 (max_length, batch_size, output_size)
		target : targetの系列 (max_length, batch_size)
	Returns:
		targetのPAD部分を0でマスクした損失
	"""
	logits_flat = logits.view(-1, logits.size(-1))
	log_probs_flat = F.log_softmax(logits_flat, -1) # (max_length*batch_size, output_size)
	target_flat = target.view(-1, 1) # (max_length*batch_size, 1)

	losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat) # (max_length*batch_size, 1)
	losses = losses_flat.view(*target.size()) # (max_length, batch_size)

	mask = (target != PAD).float()
	losses = losses * mask
	loss = losses.sum()
	return loss



def compute_loss(batch_X, batch_Y, lengths_X, model, optimizer=None, is_train=True, teacher_forcing_rate=None):

	model.train(is_train)
	use_teacher_forcing = is_train and (random.random() < teacher_forcing_rate)
	max_length = batch_Y.size(0)

	pred_Y = model(batch_X, lengths_X, max_length, batch_Y, use_teacher_forcing)
	loss = masked_cross_entropy(pred_Y.contiguous(), batch_Y.contiguous())
	# mce = nn.CrossEntropyLoss(size_average=False, ignore_index=PAD)
	# loss = mce(pred_Y.contiguous().view(-1, pred_Y.size(-1)), batch_Y.contiguous().view(-1))
	# pred_Y :(max_length, batch_size, decoder.output_size)
	# batch_Y: (max_length, batch_size)

	if is_train:
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	batch_Y = batch_Y.transpose(0, 1).contiguous().data.cpu().tolist() # (batch_size, max_length)
	pred = pred_Y.argmax(-1).data.cpu().numpy().T.tolist()

	return loss.item(), batch_Y, pred


def calc_bleu(refs, hyps):
    """
    BLEUスコアを計算する関数
    :param refs: list, 参照訳。単語のリストのリスト (例： [['I', 'have', 'a', 'pen'], ...])
    :param hyps: list, モデルの生成した訳。単語のリストのリスト (例： [['I', 'have', 'a', 'pen'], ...])
    :return: float, BLEUスコア(0~100)
    """
    refs = [[ref[:ref.index(EOS)]] for ref in refs]
    hyps = [hyp[:hyp.index(EOS)] if EOS in hyp else hyp for hyp in hyps]
    return 100 * bleu_score.corpus_bleu(refs, hyps)


def trim_eos(ids):
	if EOS in ids:
		return ids[:ids.index(EOS)]
	else:
		return ids
