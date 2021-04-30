import pandas as pd

col1 = pd.read_table("./col1.txt")
col2 = pd.read_table("./col2.txt")
print(col1)
print(col2)
merged = pd.concat([col1, col2], axis = 1)
merged.to_csv("./merge.txt", index=False, sep='\t')
print(merged)
