import re
sentence = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."

sentence = re.sub(r'[,.]', "", sentence)
words = sentence.split()
words_len = []
for word in words:
	words_len.append(len(word))

print(words_len)
