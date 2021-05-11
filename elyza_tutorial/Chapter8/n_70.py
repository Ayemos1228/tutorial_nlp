import gensim
import pickle
import numpy as np
import pandas as pd
import re
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データ読み込み
train_data = pd.read_csv("./train.txt", sep="\t")
val_data = pd.read_csv("./val.txt", sep="\t")
test_data = pd.read_csv("./test.txt", sep="\t")

with open("gensim-vc.pickle", mode="rb") as f:
	vec = pickle.load(f)

def get_feature_vec(row, vec):

	row = re.sub(r'[\W]', " ", row)
	words = row.strip().split()
	feature_vec = np.zeros(300, dtype=float)
	num_words = 0
	for word in words:
		if word in vec.key_to_index.keys():
			feature_vec += vec[word]
			num_words += 1
		else:
			continue
	feature_vec /= num_words
	return torch.tensor(feature_vec, dtype=torch.float, device=device)


train_feature = torch.stack([get_feature_vec(title, vec) for title in train_data["TITLE"].values],dim=0)
val_feature = torch.stack([get_feature_vec(title, vec) for title in val_data["TITLE"].values],dim=0)
test_feature = torch.stack([get_feature_vec(title, vec) for title in test_data["TITLE"].values],dim=0)

torch.save(train_feature, "train_feature.pt")
torch.save(val_feature, "val_feature.pt")
torch.save(test_feature, "test_feature.pt")

# label番号を取得
category_map = {
	"b": 0,
	"t": 1,
	"e": 2,
	"m": 3,
}
train_label = list(map(lambda x: category_map[x], train_data["CATEGORY"].values))
val_label = list(map(lambda x: category_map[x], val_data["CATEGORY"].values))
test_label = list(map(lambda x: category_map[x], test_data["CATEGORY"].values))
torch.save(torch.tensor(train_label, dtype=torch.long, device=device), "train_label.pt")
torch.save(torch.tensor(val_label, dtype=torch.long, device=device), "val_label.pt")
torch.save(torch.tensor(test_label, dtype=torch.long, device=device), "test_label.pt")
