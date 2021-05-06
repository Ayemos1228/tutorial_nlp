import pandas as pd

data = pd.read_json("jawiki-country.json",lines = True)
text_uk = data[data["title"] =="イギリス"]["text"].iloc[0]
print(text_uk)
