import scipy.stats
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import optuna.integration.lightgbm as lgb
import pandas as pd
import pickle

def LR_tuning(train_X, train_y, val_X, val_y, test_X, test_y,
C_list=[1e+2, 1e+1, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]):
	"""異なる正則化パラメータでロジスティック回帰モデルを学習し検証データの正解率が最も高くなるモデルを見つけ、
	その後評価データでの正解率を出力する
	Args:
		train_X 		：データ
		train_y 		：データ
		val_X 			：データ
		val_y 			：データ
		test_X 			：データ
		test_y 			：データ
		C_list  (list)	: default = [1e+2, 1e+1, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
	"""
	best_acc = 0
	best_param = None
	# learning
	for C in C_list:
		model = LogisticRegression(random_state=101, max_iter=10000, C=C).fit(train_X, train_y)
		acc = model.score(val_X, val_y)
		if acc > best_acc:
			best_acc = acc
			best_param = C

	# accuracy on test_data
	best_model = LogisticRegression(random_state=101, max_iter=10000, C=best_param).fit(train_X, train_y)
	test_acc = best_model.score(test_X, test_y)
	with open("log_reg_best.pickle", mode="wb") as f:
		pickle.dump(best_model, f)
	print(f"Best param: C = {best_param}")
	print(f"Best accuracy on val_data: {best_acc}")
	print(f"Accuracy on test_data: {test_acc}")


def LinearSVC_tuning(train_X, train_y, val_X, val_y, test_X, test_y,
C_list=[1e+2, 1e+1, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]):
	"""異なる正則化パラメータで線形サポートベクターマシンを学習し検証データの正解率が最も高くなるモデルを見つけ、
	その後評価データでの正解率を出力する
	Args:
		train_X 		：データ
		train_y 		：データ
		val_X 			：データ
		val_y 			：データ
		test_X 			：データ
		test_y 			：データ
		C_list  (list)	: default = [1e+2, 1e+1, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
	"""
	best_acc = 0
	best_param = None
	# learning
	for C in C_list:
		model = LinearSVC(random_state=101, max_iter=10000, C=C).fit(train_X, train_y)
		acc = model.score(val_X, val_y)
		if acc > best_acc:
			best_acc = acc
			best_param = C

	# accuracy on test_data
	best_model = LinearSVC(random_state=101, max_iter=10000, C=best_param).fit(train_X, train_y)
	test_acc = best_model.score(test_X, test_y)
	with open("linearsvc_best.pickle", mode="wb") as f:
		pickle.dump(best_model, f)
	print(f"Best param: C = {best_param}")
	print(f"Best accuracy on val_data: {best_acc}")
	print(f"Accuracy on test_data: {test_acc}")


def LightGBM_tuning(train_X, train_y, val_X, val_y, test_X, test_y):
	"""異なる正則化パラメータでLightGBMを学習し検証データの正解率が最も高くなるモデルを見つけ、
	その後評価データでの正解率を出力する
	Args:
		train_X 		：データ
		train_y 		：データ
		val_X 			：データ
		val_y 			：データ
		test_X 			：データ
		test_y 			：データ
	"""
	train_data = lgb.Dataset(train_X, train_y)
	val_data = lgb.Dataset(val_X, val_y, reference=train_data)
	params = {
		"objective": "multiclass",
		"boosting_type": "gbdt",
		"metric": "multi_logloss",
		"num_class": "4",
	}
	# tuning
	best_params, history = dict(), list()
	model = lgb.train(params, train_data, valid_sets=val_data, verbose_eval=-1, best_params=best_params,tuning_history=history)

	# accuracy on test_data
	# best_model = lgb.train(best_params, train_data, valid_sets=val_data, verbose_eval=-1, best_params=best_params,tuning_history=history)
	# test_acc = best_model.score(test_X, test_y)
	y_pred = model.predict(test_X)
	test_acc = accuracy_score(test_y, y_pred)
	with open("lgb_best.pickle", mode="wb") as f:
		pickle.dump(best_model, f)
	print(f"Best params: {best_params}")
	print(f"Accuracy on test_data: {test_acc}")



if __name__ == "__main__":
	# data loading...
	train_X = pd.read_csv("train.feature.txt", sep="\t")
	train_y = pd.read_csv("train.txt", sep="\t")["CATEGORY"]
	val_X = pd.read_csv("valid.feature.txt", sep="\t")
	val_y = pd.read_csv("val.txt", sep="\t")["CATEGORY"]
	test_X = pd.read_csv("test.feature.txt", sep="\t")
	test_y = pd.read_csv("test.txt", sep="\t")["CATEGORY"]

	print("----LogisticRegression----")
	LR_tuning(train_X, train_y, val_X, val_y, test_X, test_y)
	print("----LinearSVC----")
	LinearSVC_tuning(train_X, train_y, val_X, val_y, test_X, test_y)
	# print("---LightGBM---")
	# LightGBM_tuning(train_X, train_y, val_X, val_y, test_X, test_y)


