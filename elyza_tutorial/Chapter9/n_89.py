import random
from re import T
import numpy as np
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tools import load_sent, Vocab, sentence2ids
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random_state = 42
torch.manual_seed(1)

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
PAD = 0
UNK = 1
MIN_COUNT = 2

word2id = {
PAD_TOKEN: PAD,
UNK_TOKEN: UNK,
}

category_map = {
	"b": 0,
	"t": 1,
	"e": 2,
	"m": 3,
}

train_X, train_Y = load_sent("./train.txt")
test_X, test_Y = load_sent("./test.txt")
train_Y = list(map(lambda x: category_map[x], train_Y))
test_Y = list(map(lambda x: category_map[x], test_Y))


class BERTClass(nn.Module):
	def __init__(self, n_class, dropout=0.4):
		super().__init__()
		self.bert = BertModel.from_pretrained("bert-base-uncased")
		self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
		self.linear = nn.Linear(768, n_class)
		self.dropout = nn.Dropout(dropout)

	def forward(self, batch_X):
		batch_X = self.tokenizer(batch_X, padding=True, truncation=True, return_tensors="pt").to(device)
		out = self.bert(**batch_X).last_hidden_state # (batch_size, max_len, 768)
		out = out.mean(dim=1).squeeze(1) # (batch_size, 768)
		out = self.dropout(out)
		logits = self.linear(out) # (batch_size, n_class)

		return logits


def compute_loss(batch_X, batch_Y, model, criterion, optimizer=None, is_train=True):
	model.train(is_train)
	logit = model(batch_X) # (batch_size, n_class)
	loss = criterion(logit, batch_Y.to(device))
	if is_train:
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	pred = logit.argmax(dim=-1)
	n_corrects = (pred.data.cpu() == batch_Y.data).sum()
	return loss.item(), n_corrects


lr = 0.001
batch_size = 128
n_epochs = 5
dropout = 0.4
model = BERTClass(4, dropout=dropout).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.linear.parameters(), lr=lr)


train_dataloader = DataLoader(list(zip(train_X, train_Y)), batch_size=batch_size)
test_dataloader = DataLoader(list(zip(test_X, test_Y)), batch_size=batch_size)

for epoch in range(1, n_epochs+1):
	train_loss = 0.
	test_loss = 0.
	train_corrects = 0
	test_corrects = 0
	for batch_X, batch_Y in train_dataloader:
		loss, n_corrects = compute_loss(batch_X, batch_Y, model, criterion, optimizer=optimizer, is_train=True)
		train_loss += loss
		train_corrects += n_corrects
	for batch_X, batch_Y in test_dataloader:
		loss, n_corrects = compute_loss(batch_X, batch_Y, model, criterion, optimizer=optimizer, is_train=False)
		test_loss += loss
		test_corrects += n_corrects

	train_loss /= len(train_dataloader)
	test_loss /= len(test_dataloader)
	train_acc = train_corrects / len(train_X)
	test_acc = test_corrects / len(test_X)

	print('Epoch {} train_loss: {:5.2f} train_acc: {:5.2f} test_loss: {:5.2f} test_acc: {:5.2f}'.format(
		epoch, train_loss, train_acc,  test_loss, test_acc))
	print('-'*80)



