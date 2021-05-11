import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SNN(nn.Module):
	def __init__(self, feature_size=300, n_category=4):
		super().__init__()
		self.feature_size = feature_size
		self.n_category = n_category
		self.linear = nn.Linear(self.feature_size, self.n_category, bias=False)

	def forward(self, batch_X):
		"""
		Args:
			batch_X : (batch_size, feature_size=300)
		"""
		lin_X = self.linear(batch_X) # (batch_size, n_category)
		out = F.softmax(lin_X, dim=-1) # (batch_size, n_category)
		return out # (batch_size, n_category)


class DataLoaderSNN:
	def __init__(self, feature, label, batch_size):
		"""
		Args:
			feature : 特徴ベクトルの行列 (data_size, feature_size=300)
			label : 正解ラベルのベクトル (data_size, )
			batch_size : バッチサイズ
		"""
		self.feature = feature
		self.label = label
		self.batch_size = batch_size
		self.pointer = 0
		self.data_size = feature.size(0)

	def __iter__(self):
		return self

	def __next__(self):
		batch_X = self.feature[self.pointer:self.pointer + self.batch_size]
		batch_Y = self.label[self.pointer:self.pointer + self.batch_size]
		self.pointer += self.batch_size
		if self.pointer >= self.data_size:
			self.pointer = 0
			raise StopIteration

		return batch_X , batch_Y

def compute_loss(model, input, optimizer=None, is_train=True):
	"""lossを計算する関数

	Args:
		model : モデル
		input : モデルへの入力
		optimizer : optimizer
		is_train (bool, optional): 学習させるかどうか Defaults to True.
	"""
	model.train(is_train)
	pred = model(input[0])
	log_pred = torch.log(pred)
	loss = F.nll_loss(log_pred, input[1])
	if is_train:
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	return loss.item()

def calculate_accuracy(model, features, labels):
	"""正解率を計算する関数

	Args:
		model : モデル
		features : 特徴量 (data_size, n_category)
		labels : 正解ラベル (data_size, )
	"""
	model.eval()
	output = model(features) # (test_size, n_category)
	pred = torch.argmax(output, dim=-1) #(test_size)
	corrects = (pred == labels).sum()
	data_size = features.size(0)
	return (corrects / data_size).item()

batch_size = 8
lr = 0.01
n_epoch = 100

feature_train = torch.load("./train_feature.pt").to(device)
label_train = torch.load("./train_label.pt").to(device)
feature_val = torch.load("./val_feature.pt").to(device)
label_val = torch.load("./val_label.pt").to(device)
model = SNN().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr)
data_loader = DataLoaderSNN(feature_train, label_train, batch_size=batch_size)

elapsed_times = []
train_losses = []
train_accs = []
val_losses = []
val_accs = []
for epoch in range(1, n_epoch + 1):
	start_time = time.time()
	for batch, (batch_X, batch_Y) in enumerate(data_loader):
		loss = compute_loss(model, (batch_X, batch_Y), optimizer=optimizer, is_train=True)
	end_time = time.time()

	# checkpoint
	# torch.save(model.linear.weight.data.cpu().numpy(), f"./checkpoint/model_weight_epoch{epoch}.pt")
	# torch.save(optimizer.state_dict(), f"./checkpoint/optimzer_statedict_epoch{epoch}.pt")

	# 描画
	val_loss = compute_loss(model, (feature_val, label_val), is_train=False)
	train_acc = calculate_accuracy(model, feature_train, label_train)
	val_acc = calculate_accuracy(model, feature_val, label_val)
	train_losses.append(loss)
	val_losses.append(val_loss)
	train_accs.append(train_acc)
	val_accs.append(val_acc)

	fig = plt.figure(figsize=(12, 8))
	ax1 = fig.add_subplot(1, 2, 1)
	ax2 = fig.add_subplot(1, 2, 2)
	ax1.set_xlabel("epoch")
	ax1.set_ylabel("loss")
	ax2.set_xlabel("epoch")
	ax2.set_ylabel("accuracy")
	ax1.plot(train_losses, label="train")
	ax1.plot(val_losses, label="val")
	ax2.plot(train_accs, label="train")
	ax2.plot(val_accs, label="val")
	ax1.legend()
	ax2.legend()
	plt.pause(0.1)

	print(f"Epoch: {epoch}, Loss: {loss}, Elapsed time: {end_time - start_time}")
	elapsed_times.append(end_time - start_time)

print(f"Averaged elapsed time (batch_size={batch_size}): {np.mean(elapsed_times)}")



# 評価 (74)
feature_test = torch.load("./test_feature.pt").to(device)
label_test = torch.load("./test_label.pt").to(device)

## 集計
train_acc = calculate_accuracy(model, feature_train, label_train)
print("Train accuracy: {:2f}".format(train_acc))
test_acc = calculate_accuracy(model, feature_test, label_test)
print("Test accuracy : {:2f}".format(test_acc))



