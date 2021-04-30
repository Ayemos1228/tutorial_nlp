import pandas as pd

df = pd.read_table("./popular-names.txt", header=None, sep = '\t', names = ["name","sex", "num", "year"])
col2 = df["sex"]
print(col2)
col2.to_csv('./col2.txt', index = False)
