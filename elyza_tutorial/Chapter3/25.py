import pandas as pd
import re

data = pd.read_json("jawiki-country.json",lines = True)
text_uk = data[data["title"] =="イギリス"]["text"].iloc[0]
template = re.findall(r'^\{\{基礎情報 国\n(.+)^\}\}$', text_uk, flags=re.DOTALL + re.MULTILINE)
key_val_list = re.findall(r'^\|(.+?)(?=(?=\n\|)|(?=\n$))', template[0], flags=re.DOTALL + re.MULTILINE)

def create_temp_dic(key_val_list):
	template_dic = {}
	for key_val in key_val_list:
		key = key_val.split("=", 1)[0].strip(" ")
		value = key_val.split("=", 1)[1].strip(" ")
		template_dic[key] = value
	return template_dic


print(create_temp_dic(key_val_list))


