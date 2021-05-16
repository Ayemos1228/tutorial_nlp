import numpy as np
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


def load_data(file_path):
	data = []
	for line in open(file_path, encoding='utf-8'):
		words = line.strip().split()
		data.append(words)
	return data

def sentence2ids(vocab, sentence):
	ids = [vocab.word2id.get(word, UNK) for word in sentence]
	ids = [BOS] + ids + [EOS]
	return ids


class DataLoader:
	def __init__(self, X, Y, batch_size, shuffle=True):
		"""
		Args:
			X :　入力言語の文のリスト
			Y : 出力言語の文のリスト
			batch_size : バッチサイズ
			shuffle (bool, optional): シャッフルするかどうか. Defaults to True.
		"""
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
		data_X, pos_X = self.preprocess_seqs(batch_X)
		data_Y, pos_Y = self.preprocess_seqs(batch_Y)
		self.start_idx += self.batch_size

		return (data_X, pos_X), (data_Y, pos_Y)

	@staticmethod
	def preprocess_seqs(seqs):
		max_length = max([len(s) for s in seqs])
		data = [s + [PAD] * (max_length - len(s)) for s in seqs]
		positions = [[pos+1 if w != PAD else 0 for pos, w in enumerate(seq)] for seq in data]
		data_tensor = torch.tensor(data, dtype=torch.long, device=device)
		position_tensor = torch.tensor(positions, dtype=torch.long, device=device)

		return data_tensor, position_tensor


def position_encoding_init(n_position, d_pos_vec):
	"""positional_encodingの行列の初期化
	Args:
		n_position : 系列長
		d_pos_vec : 隠れ層の次元数
	"""
	# PADがある単語の位置はpos=0にする
	position_enc = np.array([
		[pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
		if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

	position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dimが偶数
	position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dimが奇数

	return torch.tensor(position_enc, dtype=torch.float)

# pe = position_encoding_init(50, 256).numpy()
# plt.figure(figsize=(16, 8))
# sns.heatmap(pe, cmap='Blues')
# plt.show()

class ScaledDotProductAttention(nn.Module):
	def __init__(self, d_model, attn_dropout=0.1):
		"""
		Args:
			d_model: 隠れ層の次元数
			attn_dropout (float, optional): ドロップアウト率. Defaults to 0.1.
		"""
		super().__init__()
		self.temper = np.power(d_model, 0.5) # スケーリング
		self.dropout = nn.Dropout(attn_dropout)

	def forward(self, q, k, v, attn_mask):
		"""
		Args:
			q : queryベクトル (n_head*batch_size, len_q, d_k)
			k : keyベクトル (n_head*batch_size, len_k, d_k)
			v : valueベクトル (n_head*batch_size, len_v, d_v)
			attn_mask : Attentionに適用するマスク (n_head*batch_size, len_q, len_k)
		Returns:
			output: 出力 (n_head*batch_size, len_q, d_v)
			attn: Attention (n_head*batch_size, len_q, len_k)
		"""
		attn = torch.bmm(q, k.transpose(1, 2)) / self.temper # (n_head*batch_size, len_q, len_k)
		attn.data.masked_fill_(attn_mask, -float('inf'))

		attn = F.softmax(attn, dim=-1)
		attn = self.dropout(attn) # (n_head*batch_size, len_q, len_k)
		output = torch.bmm(attn, v) # (n_head*batch_size, len_q, d_v)

		return output, attn

class MultiHeadAttention(nn.Module):
	def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
		super().__init__()
		self.n_head = n_head
		self.d_model = d_model
		self.d_k = d_k
		self.d_v = d_v

		# d_k = d_vとすることが多い
		self.w_qs = nn.Parameter(torch.empty([n_head, d_model, d_k], dtype=torch.float))
		self.w_ks = nn.Parameter(torch.empty([n_head, d_model, d_k], dtype=torch.float))
		self.w_vs = nn.Parameter(torch.empty([n_head, d_model, d_v], dtype=torch.float))

		nn.init.xavier_normal_(self.w_qs)
		nn.init.xavier_normal_(self.w_ks)
		nn.init.xavier_normal_(self.w_vs)

		self.attention = ScaledDotProductAttention(d_model)
		self.layer_norm = nn.LayerNorm(d_model)
		self.proj = nn.Linear(n_head*d_v, d_model) # Multiheadattentionかけた結果を元に戻す
		nn.init.xavier_normal_(self.proj.weight)

		self.dropout = nn.Dropout(dropout)

	def forward(self, q, k, v, attn_mask=None):
		"""
		Args:
			q : queryベクトル (batch_size, len_q, d_model)
			k : keyベクトル (batch_size, len_k, d_model)
			v : valueベクトル (batch_size, len_v, d_model)
			attn_mask : Attentionに適用するマスク (batch_size, len_q, len_k)
		Returns:
			outputs: 出力 (batch_size, len_q, d_model)
			attn: Attention (n_head*batch_size, len_q, len_k)
		"""
		d_k, d_v = self.d_k, self.d_v
		n_head = self.n_head

		residual = q

		batch_size, len_q, d_model = q.size()
		batch_size, len_k, d_model = k.size()
		batch_size, len_v, d_model = v.size()

		# multi-head
		# (n_head*batch_size, len_q, d_model)
		q_s = q.repeat(n_head, 1, 1)
		k_s = k.repeat(n_head, 1, 1)
		v_s = v.repeat(n_head, 1, 1)

		# headごとに変換行列を計算するために変形
		q_s = q_s.view(n_head, -1, d_model) # (n_head, batch_size*len_q, d_model)
		k_s = k_s.view(n_head, -1, d_model) # (n_head, batch_size*len_v, d_model)
		v_s = v_s.view(n_head, -1, d_model) # (n_head, batch_size*len_v, d_model)
		q_s = torch.bmm(q_s, self.w_qs) # (n_head, batch_size*len_q, d_k)
		k_s = torch.bmm(k_s, self.w_ks) # (n_head, batch_size*len_k, d_k)
		v_s = torch.bmm(v_s, self.w_vs) # (n_head, batch_size*len_v, d_v)

		# attentionを各バッチ各headごとに計算するために変形
		q_s = q_s.view(-1, len_q, d_k) # (n_head*batch_size, len_q, d_k)
		k_s = k_s.view(-1, len_k, d_k) # (n_head*batch_size, len_k, d_k)
		v_s = v_s.view(-1, len_v, d_v) # (n_head*batch_size, len_v, d_v)

		outputs, attns = self.attention(q_s, k_s, v_s, attn_mask=attn_mask.repeat(n_head, 1, 1))
		# outputs: (n_head*batch_size, len_q, d_v)
		# attns: (n_head*batch_size, len_q, len_k)

		outputs = torch.split(outputs, batch_size, dim=0) #(batch_size, len_q, d_v) * n_head
		outputs = torch.cat(outputs, dim=-1) # (batch_size, len_q, n_head*d_v)

		outputs = self.proj(outputs) # (batch_size, len_q, d_model)
		outputs = self.dropout(outputs)
		outputs = self.layer_norm(outputs + residual)

		return outputs, attns


class PositionwiseFeedForward(nn.Module):
	def __init__(self, d_hid, d_inner_hid, dropout=0.1):
		"""
		Args:
			d_hid : 隠れ層1層目の次元数
			d_inner_hid : 2層目の次元数
			dropout (float, optional): ドロップアウト率. Defaults to 0.1.
		"""
		super().__init__()
		self.w_1 = nn.Linear(d_hid, d_inner_hid) # なんでconv?
		self.w_2 = nn.Linear(d_inner_hid, d_hid)
		self.layer_norm = nn.LayerNorm(d_hid)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		"""
		Args:
			x: (batch_size, max_length, d_hid)
		Returns:
			(batch_size, max_length, d_hid)
		"""
		residual = x
		output = F.relu(self.w_1(x))
		output = self.w_2(output) # (batch_size, max_len, d_hid)
		output = self.dropout(output)

		return self.layer_norm(output + residual)


def get_attn_padding_mask(seq_q, seq_k):
	""" keyのPADの部分を0にするマスク
	Args:
		seq_q : queryの系列 (batch_size, len_q)
		seq_k : keyの系列 (batch_size, len_k)
	Returns:
		pad_attn_mask: (batch_size, len_q, len_k)
	"""
	batch_size, len_q = seq_q.size()
	batch_size, len_k = seq_k.size()
	pad_attn_mask = seq_k.data.eq(PAD).unsqueeze(1) # (batch_size, 1, len_k)
	pad_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k)

	return pad_attn_mask


def get_attn_subsequent_mask(seq):
	"""
	先の単語に対するattentionを0にするマスク
	Args:
		seq : 入力系列 (batch_size, length)
	Returns:
		subsequent_mask: (batch_size, length, length)
	"""
	attn_shape = (seq.size(1), seq.size(1))
	# 上三角行列
	subsequent_mask = torch.triu(torch.ones(attn_shape, dtype=torch.uint8, device=device), diagonal=1)
	subsequent_mask = subsequent_mask.repeat(seq.size(0), 1, 1)

	return subsequent_mask


class EncoderLayer(nn.Module):
	def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
		super().__init__()
		self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
		self.pos_ffn = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

	def forward(self, enc_input, slf_attn_mask=None):
		"""
		Args:
		enc_input: tensor, Encoderの入力,
			size=(batch_size, max_length, d_model)
		slf_attn_mask: tensor, Self Attentionの行列にかけるマスク,
			size=(batch_size, len_q, len_k)
		Returns:
		enc_output: tensor, Encoderの出力,
			size=(batch_size, max_length, d_model)
		enc_slf_attn: tensor, EncoderのSelf Attentionの行列,
			size=(n_head*batch_size, len_q, len_k)
		"""
		enc_output, enc_slf_attn = self.slf_attn(
			enc_input, enc_input, enc_input, attn_mask=slf_attn_mask)
		enc_output = self.pos_ffn(enc_output)

		return enc_output, enc_slf_attn



class Encoder(nn.Module):
	def __init__(
		self, vocab_size, max_length, n_layers=6, n_head=8, d_k=64, d_v=64,
		embedding_size=512, d_model=512, d_inner_hid=1024, dropout=0.1):

		super().__init__()
		n_position = max_length + 1
		self.max_length = max_length
		self.d_model = d_model

		# positional encoding
		self.position_enc = nn.Embedding(n_position, embedding_size, padding_idx=PAD)
		self.position_enc.weight.data = position_encoding_init(n_position, embedding_size)

		self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=PAD)

		self.layers = nn.ModuleList([
			EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
			for _ in range(n_layers)
		])

	def forward(self, src_seq, src_pos):
		"""
		Args:
			src_seq : 入力系列 (batch_size, max_length)
			src_pos : 入力系列の位置情報 (batch_size, max_length)
		Returns:
			enc_output: 出力 (batch_size, max_length, d_model)
			enc_slf_attns: Encoderの各層のAttentionのリスト [(n_head*batch_size, max_length, max_length)...] * n_layers
		"""
		enc_input = self.embedding(src_seq) # (batch_size, max_length, embedding_size)
		enc_input += self.position_enc(src_pos)

		enc_slf_attns = []
		enc_output = enc_input
		enc_slf_attn_mask = get_attn_padding_mask(src_seq, src_seq)

		for enc_layer in self.layers:
			enc_output, enc_slf_attn  = enc_layer(
				enc_output, slf_attn_mask=enc_slf_attn_mask)
			enc_slf_attns += enc_slf_attn

		return enc_output, enc_slf_attns


class DecoderLayer(nn.Module):
	def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
		super().__init__()
		# Decoder 内のAttention
		self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
		# Encoderとの間のSource-Target attenttion
		self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
		self.pos_ffn = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

	def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
		"""
		Args:
			dec_input : Decoderの入力 (batch_size, max_length, d_model)
			enc_output : Encoderの出力　(batch_size, max_length, d_model)
			slf_attn_mask ([type], optional): Decoder内のself-attentionにかけるマスク. Defaults to None.
			dec_enc_attn_mask ([type], optional): Source-Target Attentionにかけるマスク. Defaults to None.
		Returns:
			dec_output: (batch_size, max_length, d_model)
			dec_slf_attn: (n_head*batch_size, len_q, len_k)
			dec_enc_attn: (n_head*batch_size, len_q, len_k)
		"""
		# Decoder内のself-attention
		dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, attn_mask=slf_attn_mask)
		# Source-Target Attention
		dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, attn_mask=dec_enc_attn_mask)
		dec_output = self.pos_ffn(dec_output)

		return dec_output, dec_slf_attn, dec_enc_attn


class Decoder(nn.Module):
	def __init__(
		self, output_vocab, max_length, n_layers=6, n_head=8, d_k=64, d_v=64,
		embedding_size=512, d_model=512, d_inner_hid=1024, dropout=0.1):

		super().__init__()
		n_position = max_length + 1
		self.max_length = max_length
		self.d_model = d_model

		self.position_enc = nn.Embedding(n_position, embedding_size, padding_idx=PAD)
		self.position_enc.weight.data = position_encoding_init(n_position, embedding_size)

		self.embedding = nn.Embedding(output_vocab, embedding_size, padding_idx=PAD)
		self.dropout = nn.Dropout(dropout)

		self.layers = nn.ModuleList([
			DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
			for _ in range(n_layers)
		])

	def forward(self, tgt_seq, tgt_pos, src_seq, enc_output):
		"""
		Args:
			tgt_seq : 出力系列 (batch_size, tgt_length)
			tgt_pos : 出力系列の位置情報 (batch_size, tgt_length)
			src_seq : 入力系列 (batch_size, src_length)
			enc_output : (batch_size, src_length, d_model)
		Returns:
			dec_output: (batch_size, max_length, d_model)
			dec_slf_attns: [(batch_size, len_q, len_k)...] * n_layers
			dec_enc_attns: [(batch_size, len_q, len_k)...] * n_layers
		"""
		dec_input = self.embedding(tgt_seq)
		dec_input += self.position_enc(tgt_pos)

		# Decoder内のAttention用のマスク -> dec_inputのPadになっているところと、先をみないpadding
		dec_slf_attn_pad_mask = get_attn_padding_mask(tgt_seq, tgt_seq) # (batch_size, max_length, max_length)
		dec_slf_attn_sub_mask = get_attn_subsequent_mask(tgt_seq) # (batch_size, max_length, max_length)
		dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask+dec_slf_attn_sub_mask, 0) # ORをとっている

		# Source-Target Attentionのマスク -> enc_inputのPADになっているところをマスク
		dec_enc_attn_mask = get_attn_padding_mask(tgt_seq, src_seq) # (batch_size, max_length, max_length)

		dec_slf_attns, dec_enc_attns = [], []
		dec_output = dec_input
		for dec_layer in self.layers:
			dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
				dec_output, enc_output, slf_attn_mask=dec_slf_attn_mask, dec_enc_attn_mask=dec_enc_attn_mask)

			dec_slf_attns += [dec_slf_attn]
			dec_enc_attns += [dec_enc_attn]

		return dec_output, dec_slf_attns, dec_enc_attns



class Transformer(nn.Module):
	def __init__(self, input_vocab, output_vocab, max_length, n_layers=6, n_head=8,
	embedding_size=512, d_model=512, d_inner_hid=1024, d_k=64, d_v=64, dropout=0.1, proj_share_weight=True):
		"""
		Args:
			poj_share_weight (bool, optional): 出力言語の単語のembeddingと出力の写像で重みを共有する. Defaults to True.
		"""
		super().__init__()
		self.encoder = Encoder(
			input_vocab, max_length, n_layers=n_layers, n_head=n_head,
			embedding_size=embedding_size, d_model=d_model,
			d_inner_hid=d_inner_hid, dropout=dropout)
		self.decoder = Decoder(
			output_vocab, max_length, n_layers=n_layers, n_head=n_head,
			embedding_size=embedding_size, d_model=d_model,
			d_inner_hid=d_inner_hid, dropout=dropout)
		self.tgt_proj = nn.Linear(d_model, output_vocab, bias=False)
		nn.init.xavier_normal_(self.tgt_proj.weight)
		self.dropout = nn.Dropout(dropout)

		assert d_model == embedding_size, 'd_model = {}, embedding_size = {}'.format(d_model, embedding_size)

		if proj_share_weight:
			self.tgt_proj.weight = self.decoder.embedding.weight

	def get_trainable_parameters(self):
		# positional encoding以外の更新するパラメータを取得
		enc_freeze_param_ids = set(map(id, self.encoder.position_enc.parameters()))
		dec_freeze_param_ids = set(map(id, self.decoder.position_enc.parameters()))
		freeze_param_ids = enc_freeze_param_ids | dec_freeze_param_ids
		return (p for p in self.parameters() if id(p) not in freeze_param_ids)

	def forward(self, src, tgt):
		src_seq, src_pos = src
		tgt_seq, tgt_pos = tgt
		# seq: (batch_size, max_length)
		# pos: (batch_size, max_length)
		src_seq = src_seq[:, 1:] # BOSはいらない
		src_pos = src_pos[:, 1:] # BOSはいらない
		tgt_seq = tgt_seq[:, :-1] # EOSはいらない
		tgt_pos = tgt_pos[:, :-1] # EOSはいらない

		enc_output, *_ = self.encoder(src_seq, src_pos)
		dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output) # (batch_size, max_len, d_model)
		logit = self.tgt_proj(dec_output) # (batch_size, max_len, output_vocab)

		return logit

def compute_loss(batch_X, batch_Y, model, criterion, optimizer=None, is_train=True):
	model.train(is_train)
	logit = model(batch_X, batch_Y) # (batch_size, max_len, output_vocab)
	gold = batch_Y[0][:, 1:].contiguous() # (batch_size, max_len)
	loss = criterion(logit.view(-1, logit.size(2))  , gold.view(-1)) # 平たくしてCrossentropyをみる

	if is_train:
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	gold = gold.data.cpu().tolist()
	pred = logit.argmax(-1).data.cpu().tolist()

	return loss.item(), gold, pred

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


def test(model, src, max_length=20):
	# 学習済みモデルで系列を生成する
	model.eval()

	src_seq, src_pos = src
	batch_size = src_seq.size(0)
	enc_output, enc_slf_attns = model.encoder(src_seq, src_pos)

	tgt_seq = torch.full([batch_size, 1], BOS, dtype=torch.long, device=device)
	tgt_pos = torch.arange(1, dtype=torch.long, device=device)
	tgt_pos = tgt_pos.unsqueeze(0).repeat(batch_size, 1)

	# 時刻ごとに処理
	for t in range(1, max_length+1):
		dec_output, dec_slf_attns, dec_enc_attns = model.decoder(
			tgt_seq, tgt_pos, src_seq, enc_output)
		dec_output = model.tgt_word_proj(dec_output)
		out = dec_output[:, -1, :].max(dim=-1)[1].unsqueeze(1)
		# 自身の出力を次の時刻の入力にする
		tgt_seq = torch.cat([tgt_seq, out], dim=-1)
		tgt_pos = torch.arange(t+1, dtype=torch.long, device=device)
		tgt_pos = tgt_pos.unsqueeze(0).repeat(batch_size, 1)

	return tgt_seq[:, 1:], enc_slf_attns, dec_slf_attns, dec_enc_attns
