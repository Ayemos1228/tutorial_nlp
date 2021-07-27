import gensim
import pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

with open("gensim-vc.pickle", mode="rb") as f:
    vec = pickle.load(f)

human_rate = []
w2v_rate = []
with open("./wordsim353/combined.csv", "r") as f:  # どうでもいいけど、read_csvの方が良くない？
    lines = f.readlines()
    for line in lines:
        if line.startswith("Word"):
            continue
        data = line.split(",")
        human_rate.append(data[2])
        w2v_rate.append(vec.similarity(data[0], data[1]))


correlation, pvalue = spearmanr(w2v_rate, human_rate)
print(f"Spearman Correlation: {correlation}")
