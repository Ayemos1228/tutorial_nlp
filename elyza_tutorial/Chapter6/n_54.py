from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd


def get_accuracy(model_path, X, y):
	"""モデルのpickleファイルへのパスと特徴りょうを受け取って、
	accuracyを返す確率

	Args:
		model_path (path)	: pickleファイルのパス
		X					: 特徴量
		y					: 正解ラベル
	Returns:
		accuracy			: accuracy
	"""
	with open(model_path, mode="rb") as f:
		model = pickle.load(f)
	accuracy = model.score(X, y)

	return accuracy


if __name__ == "__main__":
	train_X = pd.read_csv("train.feature.txt", sep="\t")
	train_y = pd.read_csv("train.txt", sep="\t")["CATEGORY"]
	test_X = pd.read_csv("test.feature.txt", sep="\t")
	test_y = pd.read_csv("test.txt", sep="\t")["CATEGORY"]

	train_acc = get_accuracy("maxf_5000.pickle", train_X, train_y)
	test_acc = get_accuracy("maxf_5000.pickle", test_X, test_y)

	print(f"train_acc = {train_acc}, test_acc = {test_acc}")
