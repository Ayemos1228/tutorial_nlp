import pandas as pd

df = pd.read_table("./popular-names.txt", header=None, \
	sep = '\t', names= ["name", "sex", "num", "year"])
print(df["name"].value_counts())
