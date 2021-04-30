import re
str = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."

str = re.sub(r'[,.]', "", str)
list = str.split(" ")
dict = {}
for item in list:
	dict[item] = len(item)

print(dict)
