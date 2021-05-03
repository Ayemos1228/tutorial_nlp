import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Hiragino Maru Gothic Pro"]

def LR_regularization_tuning(train_X, train_y, val_X, val_y, test_X, test_y,
C_list=[1e+2, 1e+1, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]):
	"""異なる正則化パラメータでロジスティック回帰モデルを学習し，
	学習データ，検証データ，および評価データ上の正解率を求めてプロットする関数
	Args:
		train_X 		：データ
		train_y 		：データ
		val_X 			：データ
		val_y 			：データ
		test_X 			：データ
		test_y 			：データ
		range  (list)	: default = [1e+2, 1e+1, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
	"""
	train_acc = []
	val_acc = []
	test_acc = []
	# learning
	for C in C_list:
		model = LogisticRegression(random_state=101, max_iter=10000, C=C).fit(train_X, train_y)
		train_acc.append(model.score(train_X, train_y))
		val_acc.append(model.score(val_X, val_y))
		test_acc.append(model.score(test_X, test_y))

	# plot
	plt.plot(C_list, train_acc, "r", label="train")
	plt.plot(C_list, val_acc, "b", label="val")
	plt.plot(C_list, test_acc, "g", label="test")
	plt.xticks(C_list, C_list)
	plt.xlabel("C")
	plt.xscale("log")
	plt.ylabel("Accuracy")
	plt.legend(bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=1, fontsize=10)
	plt.show()


if __name__ == "__main__":
	train_X = pd.read_csv("train.feature.txt", sep="\t")
	train_y = pd.read_csv("train.txt", sep="\t")["CATEGORY"]
	val_X = pd.read_csv("valid.feature.txt", sep="\t")
	val_y = pd.read_csv("val.txt", sep="\t")["CATEGORY"]
	test_X = pd.read_csv("test.feature.txt", sep="\t")
	test_y = pd.read_csv("test.txt", sep="\t")["CATEGORY"]

	LR_regularization_tuning(train_X, train_y, val_X, val_y, test_X, test_y)

