from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd

def get_prediction(model_path, vectorizer_path, title):
	"""モデルのpickleファイルへのパスとタイトルを受け取って、
	カテゴリとその予測確率を返す関数

	Args:
		model_path (path)	: pickleファイルのパス
		title (string)		: 記事見出し
	Returns:
		predicted_category	: 予測カテゴリ
		prob				: 予測確率
	"""
	with open(model_path, mode="rb") as f:
		model = pickle.load(f)
	with open(vectorizer_path, mode="rb") as g:
		vectorizer = pickle.load(g)
	X = vectorizer.transform([title]).toarray()
	predicted_category = model.predict(X)
	prob = model.predict_proba(X)

	return predicted_category, prob


if __name__ == "__main__":
	category, prob = get_prediction("maxf_5000.pickle", "tf-idfvectorizer.pickle", "Britain's Top Credit Rating Secured at S&P After Outlook Raised")
	print(f'Correct category = b, Predicted category = {category}, prob = {prob}')

