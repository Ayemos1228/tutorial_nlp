import gensim
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage
from n_67 import extract_country_name


with open("gensim-vc.pickle", mode="rb") as f:
	vec = pickle.load(f)

country_list = extract_country_name("analogy.txt")
country_vec = [vec[country] for country in country_list]

cluster = linkage(country_vec, method="ward")
dendrogram(cluster, labels=country_list)
plt.show()
