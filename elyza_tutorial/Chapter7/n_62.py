import gensim
import pickle


def print_simwords(word_vec, word, n=10):
	"""wordとコサイン類似度が高いn個の単語を出力する関数

	Args:
		word_vec 			: 単語ベクトルの集合
		word (string)		: 類似度を計算する単語
		n (int, optional)	: n個出力
	"""
	sim_words = vec.most_similar(positive=[word], topn=n)
	print(f"---top-10 most similar words with {word}---")
	for sim_word in sim_words:
		print(f"word: {sim_word[0]}")
		print(f"similarity: {sim_word[1]}")
		print("-"*20)


if __name__ == "__main__":
	with open("gensim-vc.pickle", mode="rb") as f:
		vec = pickle.load(f)
	print_simwords(vec, "United_States")



