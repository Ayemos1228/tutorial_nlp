import gensim
import pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans


def extract_country_name(analogy_text_path):
	country_set = set()
	country_column_num = -1
	with open(analogy_text_path, mode="r") as f:
		lines = f.readlines()
		for line in lines:
			if ":" in line:
				if "countries" in line or "capital-world" in line:
					country_column_num= 1
					continue
				elif "currency" in line or "nationality-adjective" in line:
					country_column_num = 0
				else:
					country_column_num= -1
					continue
			else:
				if country_column_num != -1:
					countries = line.rstrip("\n").split(" ")
					country_set.add(countries[country_column_num])

	return list(country_set)


if __name__ == "__main__":
	with open("gensim-vc.pickle", mode="rb") as f:
		vec = pickle.load(f)
	country_list = extract_country_name("analogy.txt")
	country_vec = [vec[country] for country in country_list]
	model = KMeans(n_clusters=5, random_state=101).fit(country_vec)
	for c in range(5):
		print(f"Cluster: {c}")
		print(", ".join([country_list[i] for i in range(len(country_list)) if model.labels_[i] == c]))
		print("-"*20)
