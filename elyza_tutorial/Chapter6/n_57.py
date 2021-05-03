from sklearn.metrics import classification_report
import pickle
import pandas as pd

with open("maxf_5000.pickle", mode="rb") as f:
	model = pickle.load(f)

print(model.coef_.shape)
