import pandas as pd

df = pd.read_table("./popular-names.txt", header=None, sep = '\t')
print(len(df))
