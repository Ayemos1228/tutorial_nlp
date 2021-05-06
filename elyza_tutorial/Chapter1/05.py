def n_gram(seq, n):
	ans = []
	tmp = []
	for i in range (len(seq) - n + 1):
		tmp = []
		for j in range (n):
			tmp.append(seq[i + j])
		ans.append(tmp)
	return (ans)

seq = "I am an NLPer"
list = seq.split(" ")
ret = n_gram(list, 2)
print(ret) # 文字bi-gramも欲しいらしい
