import re
str = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
str = re.sub(r'[,.]', "", str)
list = str.split(" ")

dict = {}
for i, item in enumerate(list):
	if i in [0, 4, 5, 6, 7, 8, 14, 15, 18]:
		dict[item[:1]] = i
	else:
		dict[item[:2]] = i
print(dict)
