import sys
import pandas as pd

df = pd.read_table("./popular-names.txt", header=None, \
	sep = '\t', names= ["name", "sex", "num", "year"])
n = int(sys.argv[1])
num_row = len(df) // n
for idx in range(n):
	df.iloc[num_row * idx:num_row * (idx + 1)].to_csv(f'split_{idx}.txt', sep = "\t", index=False)
