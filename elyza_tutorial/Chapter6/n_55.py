from sklearn.metrics import confusion_matrix
import pickle
import pandas as pd

with open("maxf_5000.pickle", mode="rb") as f:
	model = pickle.load(f)

# get true y and predicted y
train_X = pd.read_csv("train.feature.txt", sep="\t")
test_X = pd.read_csv("test.feature.txt", sep="\t")
train_y_true= pd.read_csv("train.txt", sep="\t")["CATEGORY"]
train_y_pred = model.predict(train_X)
test_y_true = pd.read_csv("test.txt", sep="\t")["CATEGORY"]
test_y_pred = model.predict(test_X)

# confusion matrix
print("----train_data----")
print(confusion_matrix(train_y_true, train_y_pred))
print("----test_data----")
print(confusion_matrix(test_y_true, test_y_pred))

