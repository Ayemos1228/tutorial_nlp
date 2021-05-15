import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tools import load_data, Vocab, sentence2ids, compute_loss_cnn, get_embedding, CNN, DataLoaderCNN, TextCNN

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

train_X, train_Y = load_data("./train.txt")
# valid_X, valid_Y = load_data("./val.txt")
test_X, test_Y = load_data("./test.txt")

vocab = Vocab(word2id)
vocab.build_vocab(train_X, MIN_COUNT)

train_X = [sentence2ids(vocab, sent) for sent in train_X]
# valid_X = [sentence2ids(vocab, sent) for sent in valid_X]
test_X = [sentence2ids(vocab, sent) for sent in test_X]

train_Y = list(map(lambda x: category_map[x], train_Y))
test_Y = list(map(lambda x: category_map[x], test_Y))


vocab_size = len(vocab.id2word)
batch_size = 128
embedding_size = 300
kernel_num = 25
kernel_size = 4
kernel_sizes = [3, 4, 5]
lr = 0.01
num_epochs = 40
static = True
dropout = 0.4
ckpt_path = "cnn.pth"


# 事前学種済みembeddingを使用
embedding = get_embedding(vocab, "./w2v.pickle")
# model = CNN(vocab_size, embedding_size, kernel_num, kernel_size, embedding).to(device)
model = TextCNN(vocab_size, embedding_size, 4, kernel_num, kernel_sizes, dropout, static, embedding).to(device)
optimizer = optim.SGD(model.parameters(), lr=lr)
train_dataloader = DataLoaderCNN(train_X, train_Y, batch_size)
test_dataloader = DataLoaderCNN(test_X, test_Y, batch_size)


best_test_loss = 0.
for epoch in range(1, num_epochs + 1):
	train_loss = 0.
	test_loss = 0.
	train_corrects = 0
	test_corrects = 0

	for batch in train_dataloader:
		batch_X, batch_Y = batch
		loss, n_corrects = compute_loss_cnn(
			batch_X, batch_Y, model, optimizer, is_train=True)
		train_loss += loss
		train_corrects += n_corrects

	for batch in test_dataloader:
		batch_X, batch_Y = batch
		loss, n_corrects = compute_loss_cnn(batch_X, batch_Y, model, is_train=False)
		test_loss += loss
		test_corrects += n_corrects

	train_loss = train_loss / len(train_dataloader.data)
	test_loss = np.sum(test_loss) / len(test_dataloader.data)
	train_acc = train_corrects / len(train_dataloader.data)
	test_acc = test_corrects / len(test_dataloader.data)

	if test_loss < best_test_loss:
		state_dict = model.state_dict()
		torch.save(state_dict, ckpt_path)
		best_test_loss = test_loss

	print('Epoch {}: train_loss: {:5.3f} train_acc: {:5.2f} test_loss: {:5.3f} test_acc: {:5.2f}'.format(
			epoch, train_loss, train_acc, test_loss, test_acc))
	print('-'*80)
