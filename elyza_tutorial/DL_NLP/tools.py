import MeCab

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


def tokenize(sentence):
	"""日本語の文を形態素の列に分割する関数

	Args:
		sentence (str) : 日本語の文
	Returns:
		tokenized_sentence (list of str) : 形態素のリスト
	"""
	tagger = MeCab.Tagger("-Ochasen")
	node = tagger.parse(sentence)
	node = node.split("\n")
	tokenized_sentence = []
	for i in range(len(node)):
		feature = node[i].split("\t")
		if feature[0] == "EOS":
			# 文が終わったら終了
			break
		# 分割された形態素を追加
		tokenized_sentence.append(feature[0])
	return tokenized_sentence


def load_data(path):
	"""テキストを読み込んでMecabで解析する関数

	Args:
		path (str): テキストのパス
	Returns:
		text (list of list of str) : 各文がトークナイズされたテキスト
	"""
	text = []
	with open(path, "r") as f:
		for line in f:
			line = line.strip()
			line = tokenize(line)
			text.append(line)
	return text

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


def compute_loss(model, input, optimizer=None, is_train=True):
	"""lossを計算する関数

	Args:
		model : モデル
		input : モデルへの入力
		optimizer : optimizer
		is_train (bool, optional): 学習させるかどうか Defaults to True.
	"""

	model.train(is_train)
	loss = model(*input)

	if is_train:
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	return loss.item()
