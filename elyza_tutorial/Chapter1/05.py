def n_gram(seq, n):
	ans = []
	tmp = []
	for i in range (len(seq) - n + 1):
		tmp = []
		for j in range (n):
			tmp.append(seq[i + j])
		ans.append(tmp)
	return (ans)

if __name__ == "__main__":
	seq = "I am an NLPer"
	words = seq.split(" ")
	word_bigram = n_gram(words, 2)
	char_bigram = n_gram(seq, 2)
	print("---word_bigram---")
	print(word_bigram)
	print("---char_bigram---")
	print(char_bigram)
