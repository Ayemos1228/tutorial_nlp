import gensim
import pickle
import numpy as np

def AistoBasCistoX(word_vec, A, B, C):
	"""A, B, C三つの単語の単語ベクトルを受け取って、
	A - B + Cで計算できるベクトルと最も類似度の高い10語を出力する関数

	Args:
		word_vec	: 単語ベクトルの集合
		A (string)	: 単語
		B (string)	: 単語
		C (string)	: 単語
	"""
	vec_A = word_vec[A]
	vec_B = word_vec[B]
	vec_C = word_vec[C]
	vec_X = vec_A - vec_B + vec_C

	sim_words = word_vec.similar_by_vector(vec_X)
	for sim_word in sim_words:
		print(f"word: {sim_word[0]}")
		print(f"similarity: {sim_word[1]}")
		print("-"*20)


if __name__ == "__main__":
	with open("gensim-vc.pickle", mode="rb") as f:
		vec = pickle.load(f)
	AistoBasCistoX(vec, "Spain", "Madrid", "Athens")
	# print(vec.most_similar(positive=["Spain", "Athens"], negative=["Madrid"]))

