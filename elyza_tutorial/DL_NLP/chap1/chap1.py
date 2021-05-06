import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torchvision
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

in_dim  = 784
hid_dim = 300
out_dim = 10
lr = 0.001
batch_size = 32
num_epochs = 4

# データセットの読み込み
train_data = torchvision.datasets.FashionMNIST(
    './data/fashion-mnist',
    transform=torchvision.transforms.ToTensor(),
    train=True,
    download=True)
test_data = torchvision.datasets.FashionMNIST(
    './data/fashion-mnist',
    transform=torchvision.transforms.ToTensor(),
    train=False,
    download=True)

# データローダの定義
train_data_loader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True)
test_data_loader = torch.utils.data.DataLoader(
    dataset=test_data,
    batch_size=batch_size,
    shuffle=False)


# モデルの定義
class MLP(nn.Module):
	def __init__(self, in_dim, hid_dim, out_dim):
		super(MLP, self).__init__()
		self.linear_1 = nn.Linear(in_dim, hid_dim)
		self.linear_2 = nn.Linear(hid_dim, out_dim)

	def forward(self, x):
		x = F.relu(self.linear_1(x))
		x = F.log_softmax(self.linear_2(x), dim=-1)
		# x = self.linear_2(x)
		return x

mlp = MLP(in_dim, hid_dim, out_dim)
optimizer = optim.Adam(mlp.parameters(), lr=lr)
criterion = nn.NLLLoss()
# criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
	loss_train = []
	loss_test = []
	pred_train = []
	pred_test = []
	gold_train = []
	gold_test = []

	# 学習
	mlp.train()
	for x, y in train_data_loader:
		x = x.to(device)
		y = y.to(device)
		x = x.view(x.size(0), -1)
		y_pred = mlp.forward(x)
		loss = criterion(y_pred, y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# 記録
		y_pred = y_pred.argmax(1).tolist()
		pred_train.extend(y_pred)
		gold_train.extend(y.tolist())
		loss_train.append(loss.item())

	# 予測
	mlp.eval()
	for x, y in test_data_loader:
		x = x.to(device)
		y = y.to(device)
		x = x.view(x.size(0), -1)
		y_pred = mlp.forward(x)
		loss = criterion(y_pred, y)

		# 記録
		y_pred = y_pred.argmax(1).tolist()
		pred_test.extend(y_pred)
		gold_test.extend(y.tolist())
		loss_test.append(loss.item())

	print('EPOCH: {}, Train [Loss: {:.3f}, F1: {:.3f}], test [Loss: {:.3f}, F1: {:.3f}]'.format(
	epoch,
	np.mean(loss_train),
	f1_score(gold_train, pred_train, average='macro'),
	np.mean(loss_test),
	f1_score(gold_test, pred_test, average='macro')
))

# 予測結果を保存
SUBMISSION_PATH = 'submission.csv'
with open(SUBMISSION_PATH, 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(pred_test)
