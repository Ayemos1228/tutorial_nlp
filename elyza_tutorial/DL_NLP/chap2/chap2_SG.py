import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torchvision
from sklearn.metrics import f1_score
import sys
sys.path.append("../")
from tools import load_data, Vocab, sentence2ids, compute_loss
from tools_cbow import CBOW,DataLoaderCBOW
from tools_skipgram import SkipGram, SkipGramNS, DataLoaderSG, DataLoaderSGNS
from scipy.stats import spearmanr, rankdata

# device設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
PAD = 0
UNK = 1
word2id = {
	PAD_TOKEN: PAD,
	UNK_TOKEN: UNK,
}
MIN_COUNT = 3

# # vocab準備
with open("./text8", "r") as f:
	line = f.readline()
	text = line.strip().split()

text = text[:1000000]
vocab = Vocab(word2id=word2id)
vocab.build_vocab([text], min_count=MIN_COUNT)
id_text = [sentence2ids(vocab, text)]

batch_size = 64
n_batches = 1500
vocab_size = len(vocab.word2id)
embedding_size = 300

# モデル
weights = np.power([0, 0] + list(vocab.raw_vocab.values()), 0.75)
weights = weights / weights.sum()
data_loader = DataLoaderSGNS(id_text, batch_size, weights=weights)
sgns = SkipGramNS(vocab_size, embedding_size)
optimizer = optim.Adam(sgns.parameters())

for batch_idx, (batch_X, batch_Y, batch_N) in enumerate(data_loader):
	loss = compute_loss(sgns, (batch_X, batch_Y, batch_N), optimizer=optimizer, is_train=True)
	if batch_idx % 100 == 0:
		print(f"batch: {batch_idx}, loss: {loss}")
	if batch_idx >= n_batches:
		break

# データ書き出し
torch.save(sgns.in_embedding.weight.data.cpu().numpy(), "./skipgramNS.pth")
emb = torch.load("./skipgramNS.pth")
embedding_matrix = emb
word_pairs = []
with open("./sample_submission.csv", "r") as fin:
    for line in fin:
        line = line.strip().split(",")
        word1 = line[0]
        word2 = line[1]
        word_pairs.append([word1, word2])

pred_scores = []
for pair in word_pairs:
    w1 = embedding_matrix[vocab.word2id[pair[0]]]
    w2 = embedding_matrix[vocab.word2id[pair[1]]]
    score = np.dot(w1, w2)/(np.linalg.norm(w1, ord=2) * np.linalg.norm(w2, ord=2) + 1e-10)
    pred_scores.append(score)

with open("./submission.csv", "w") as fout:
    for pair, score in zip(word_pairs, pred_scores):
        fout.write(pair[0] + "," + pair[1] + "," + str(score) + "\n")

# 評価
human_rates = []
with open("./wordsim.csv", "r") as g:
	for line in g:
		line = line.strip().split(",")
		human_rate = line[2]
		human_rates.append(float(human_rate))


correlation, pvalue = spearmanr(pred_scores, human_rates)
print("Spearman correlation with human rates:", correlation)
