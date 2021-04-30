import sys
import pandas as pd

df = pd.read_table("./popular-names.txt", header=None, sep = '\t', names= ["name", "sex", "num", "year"])

n = int(sys.argv[1])
print(df.iloc[-n:, :])
