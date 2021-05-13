from elyza_tutorial.DL_NLP.tools import ids2sentence
from elyza_tutorial.DL_NLP.chap3.tools_seq import calc_bleu
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
from tools_seq import load_data, sentence2ids, DataLoader, EncoderDecoder, compute_loss, trim_eos

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

# データ読み込み
train_E = load_data("./data/train.en")
train_J = load_data("./data/train.ja")
# 縮小
train_E = train_E[:len(train_E) // 2]
train_J = train_J[:len(train_J) // 2]

# 辞書作成
vocab_E = Vocab(word2id=word2id)
vocab_J = Vocab(word2id=word2id)
vocab_E.build_vocab(train_E, min_count=MIN_COUNT)
vocab_J.build_vocab(train_J, min_count=MIN_COUNT)

# 訓練データ検証データに分割, idに変換
train_E, valid_E, train_J, valid_J = train_test_split(train_E, train_J, test_size=0.2, random_state=random_state)
train_E = [sentence2ids(vocab_E, sent) for sent in train_E]
train_J = [sentence2ids(vocab_J, sent) for sent in train_J]
valid_E = [sentence2ids(vocab_E, sent) for sent in valid_E]
valid_J = [sentence2ids(vocab_J, sent) for sent in valid_J]

# ハイパーパラメータの設定
num_epochs = 5
batch_size = 64
lr = 0.01
teacher_forcing_rate = 0.2  # Teacher Forcingを行う確率
ckpt_path = 'model.pth'  # 学習済みのモデルを保存するパス

vocab_size_E = len(vocab_E.id2word)
vocab_size_J = len(vocab_J.id2word)
hidden_size = 256

train_dataloader = DataLoader(train_E, train_J, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_E, valid_J, batch_size=batch_size, shuffle=False)

model = EncoderDecoder(vocab_size_E, vocab_size_J, hidden_size=hidden_size)
optimizer = optim.Adam(model.parameters(), lr=lr)


# 学習
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
		loss, gold, pred = compute_loss(
			batch_X, batch_Y, lengths_X, model, optimizer, is_train=True, teacher_forcing_rate=teacher_forcing_rate)
		train_loss += loss
		train_refs += gold
		train_hyps += pred

	for batch in valid_dataloader:
		batch_X, batch_Y, lengths_X = batch
		loss, gold, pred = compute_loss(batch_X, batch_Y, lengths_X, model, is_train=False)
		valid_loss += loss
		valid_refs += gold
		valid_hyps += pred

	train_loss = np.sum(train_loss) / len(train_dataloader.data)
	valid_loss = np.sum(valid_loss) / len(valid_dataloader.data)
	train_bleu = calc_bleu(train_refs, train_hyps)
	valid_bleu = calc_bleu(valid_refs, valid_hyps)

	if valid_bleu > best_valid_bleu:
		state_dict = model.state_dict()
		torch.save(state_dict, ckpt_path)
		best_valid_bleu = valid_bleu

	print('Epoch {}: train_loss: {:5.2f}  train_bleu: {:2.2f}  valid_loss: {:5.2f}  valid_bleu: {:2.2f}'.format(
			epoch, train_loss, train_bleu, valid_loss, valid_bleu))
	print('-'*80)



# 評価
ckpt = torch.load(ckpt_path)
model.load_state_dict(ckpt)
model.eval()


test_X = load_data('./data/dev.en')
test_Y = load_data('./data/dev.ja')

test_X = [sentence2ids(vocab_E, sentence) for sentence in test_X]
test_Y = [sentence2ids(vocab_J, sentence) for sentence in test_Y]

test_dataloader = DataLoader(test_X, test_Y, batch_size=1, shuffle=False)

# 生成
batch_X, batch_Y, lengths_X = next(test_dataloader)
sentence_X = ' '.join(ids2sentence(vocab_E, batch_X.data.cpu().numpy()[:-1, 0]))
sentence_Y = ' '.join(ids2sentence(vocab_J, batch_Y.data.cpu().numpy()[:-1, 0]))

output = model(batch_X, lengths_X, max_length=20)
output = output.argmax(-1).view(-1).data.cpu().tolist()
output_sentence = ' '.join(ids2sentence(vocab_J, trim_eos(output)))
print('src: {}'.format(sentence_X))
print('tgt: {}'.format(sentence_Y))
print('out: {}'.format(output_sentence))


# Bleu
refs_list = []
hyps_list = []

for batch in test_dataloader:
	batch_X, batch_Y, lengths_X = batch
	pred_Y = model(batch_X, lengths_X, max_length=20)
	pred = pred_Y.argmax(-1).view(-1).data.cpu().tolist()
	refs = batch_Y.view(-1).data.cpu().tolist()
	refs_list.append(refs)
	hyps_list.append(pred)
bleu = calc_bleu(refs_list, hyps_list)
print("bleu:", bleu)
