import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("./NewsAggregatorDataset/newsCorpora.csv", sep="\t", names=["ID", "TITLE", "URL", "PUBLISHER", "CATEGORY","STORY", "HOSTNAME", "TIMESTAMP"])
df = df[df["PUBLISHER"].isin(["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"])]

X = df["TITLE"]
y = df["CATEGORY"]

# train_test_val split
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size = 0.2, random_state=101)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size = 0.5, random_state=101)

# concat
train_data = pd.concat([X_train, y_train],axis=1)
val_data = pd.concat([X_val, y_val],axis=1)
test_data = pd.concat([X_test, y_test],axis=1)

# data status
print("----train_data----")
print(train_data["CATEGORY"].value_counts())
print("----val_data----")
print(val_data["CATEGORY"].value_counts())
print("----test_data----")
print(test_data["CATEGORY"].value_counts())

# write data to csv
train_data.to_csv("train.txt", sep="\t",index=False)
val_data.to_csv("val.txt", sep="\t",index=False)
test_data.to_csv("test.txt", sep="\t",index=False)

