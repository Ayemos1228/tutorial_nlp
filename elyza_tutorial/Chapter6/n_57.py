import pickle
import numpy as np
import pandas as pd

def get_top_bottom_10features(model, features, n=10):
	"""logistic回帰モデルの特徴量重みの高い（低い）トップnをカテゴリごとに表示する関数

	Args:
		model : logistic回帰モデル
		features: 特徴量
		n (int, optional): n個出力
	"""
	for idx, category in enumerate(model.classes_):
		if idx < len(model.classes_):
			print(f"category: {category}")
			sorted_coef_idx = np.argsort(abs(model.coef_[idx]))
			top_n = features[sorted_coef_idx[-10:]]
			bottom_n = features[sorted_coef_idx[:10]]
			print(f"Top {n}: {top_n}")
			print(f"Bottom {n}: {bottom_n}")
			print("-"*20)


if __name__ == "__main__":
	with open("maxf_5000.pickle", mode="rb") as f:
		model = pickle.load(f)
	train_features = pd.read_csv("train.feature.txt", sep="\t").columns.values
	get_top_bottom_10features(model, train_features)

