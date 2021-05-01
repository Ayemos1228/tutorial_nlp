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

# ここまで25の内容

def del_markup(dic):
	for key, value in dic.items():
		dic[key] = re.sub(r'(?:(?:\'{2,3})|(?:\'{5}))(.+?)(?:(?:\'{2,3})|(?:\'{5}))', r'\1', value)

def del_link(dic):
	"""
	内部リンクの除去を行う。初めに'|'を含むものを処理してその後にそのほかを処理する。
	"""
	for key, value in dic.items():
		trimmed = re.sub(r'\[\[(?!ファイル)(?:[^\]]+?)\|([^\]]+?)\]\]', r'\1', value)
		trimmed = re.sub(r'\[\[(?!ファイル)(.+?)\]\]', r'\1', trimmed)
		dic[key] = trimmed

def del_reference(dic):
	"""
	<ref> .... </ref>の形のreferenceを除去する。
	"""
	for key, value in dic.items():
		trimmed = re.sub(r'<ref>.+</ref>', '', value, flags=re.DOTALL)
		trimmed = re.sub(r'<ref.+$', '', trimmed, flags=re.MULTILINE)
		dic[key] = trimmed

def del_num_fmt(dic):
	"""
	{{0}}などを消す
	"""
	for key, value in dic.items():
		trimmed = re.sub(r'\{\{[0-9]+\}\}', '', value)
		dic[key] = trimmed

def replace_br(dic):
	"""
	<br /> (箇条書き)をスペースで置き換える
	"""
	for key, value in dic.items():
		trimmed = re.sub(r'<br />', ' ', value)
		dic[key] = trimmed

def del_lang_spec(dic):
	"""
	言語指定がある部分(e.g. {{lang|fr|Dieu et mon droit}})
	は最後の部分のみを抜き出す（e.g. Dieu et mon droit)
	"""
	for key, value in dic.items():
		trimmed = re.sub(r'\{\{lang\|.+\|(.+?)\}\}', r'\1', value)
		dic[key] = trimmed


template_dic = create_temp_dic(key_val_list)
del_markup(template_dic)
del_link(template_dic)
del_reference(template_dic)
del_num_fmt(template_dic)
replace_br(template_dic)
del_lang_spec(template_dic)

for key, value in template_dic.items():
	print(f'{key}: {value}')
print()

