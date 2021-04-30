import pandas as pd

df = pd.read_table("./popular-names.txt", header=None, sep = '\t', names = ["name","sex", "num", "year"])
col1 = df["name"]
print(col1)
col1.to_csv('./col1.txt', index = False)
