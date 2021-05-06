import re
str = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
# str, list, dictは組み込み関数なので変数名にしちゃダメ
str = re.sub(r'[,.]', "", str)
list = str.split(" ") # .split()でいい
dict = {}
for item in list:
	dict[item] = len(item)

print(dict) # listを返して欲しいらしい
