
import pandas as pd 
import re


#20
data = pd.read_json("jawiki-country.json",lines = True)
text_uk = data[data["title"] =="イギリス"]["text"].iloc[0]

# 21
# re.findall(r"^.*\[\[Category.+$", text_uk, flags=re.MULTILINE)

# 22
# re.findall(r"^.*\[\[Category:(\w+)",text_uk,flags=re.MULTILINE)

# 23 
# section_list = re.findall(r"(=+)(\w+)=+",text_uk,flags=re.MULTILINE)
# ans_dic = {}
# for section in section_list:
#     ans_dic[section[1]] =len(section[0]) -1
# ans_dic;

# 24 
# re.findall(r"ファイル:([^|\]]*)",text_uk);

#25
ans = re.findall(r"基礎情報[\w\W]*?",text_uk, flags=re.MULTILINE)
print(ans)
text_uk

