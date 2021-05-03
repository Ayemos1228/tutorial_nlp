from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd

train_X = pd.read_csv("train.feature.txt", sep="\t")
train_y = pd.read_csv("train.txt", sep="\t")["CATEGORY"]

model = LogisticRegression(random_state=101, max_iter = 10000).fit(train_X, train_y)
with open("maxf_5000.pickle", mode="wb") as f:
	pickle.dump(model, f)
