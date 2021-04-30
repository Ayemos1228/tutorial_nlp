import pandas as pd
import re

def get_sec_level(str):
	"""a function to get section level

	Args:
		str (string): a string with multiple "="'s and a section name

	Returns:
		[int]: section level
	"""
	level = -1
	i = 0
	while (str[i] == '='):
		level += 1
		i += 1
	return (level)

data = pd.read_json("jawiki-country.json",lines = True)
text_uk = data[data["title"] =="イギリス"]["text"].iloc[0]

sec_list = re.findall(r"(=+[\w]+=+)$",text_uk,flags=re.MULTILINE)
sec_name_list = [sec_str.strip("=") for sec_str in sec_list]
section_dic = {}

for sec_str in sec_list:
	section_dic[sec_str.strip("=")] = get_sec_level(sec_str)
print(section_dic)
