import gensim
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from n_67 import extract_country_name


with open("gensim-vc.pickle", mode="rb") as f:
	vec = pickle.load(f)

country_list = extract_country_name("analogy.txt")
country_vec = [vec[country] for country in country_list]

tsne = TSNE(n_components=2, random_state=101)
embed = tsne.fit_transform(country_vec)
plt.scatter(embed[:, 0], embed[:, 1], s=10)
for i, label in enumerate(country_list):
    plt.text(embed[i, 0], embed[i, 1],label, size=5)
plt.show()
