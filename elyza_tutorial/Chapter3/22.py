import pandas as pd
import re

data = pd.read_json("jawiki-country.json", lines=True)
text_uk = data[data["title"] == "イギリス"]["text"].iloc[0]
print(
    re.findall(r"^.*\[\[Category:(\w+)", text_uk, flags=re.MULTILINE)
)  # printが閉じられてないのでエラーになる
# どうでもいいけど"欧州連合加盟国|元"の"|元"はそのまま取っておきたい感はある
