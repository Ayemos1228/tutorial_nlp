import gensim
import pickle
import numpy as np



def cos_sim(vec1, vec2):
	return (np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

if __name__ == "__main__":
	with open("gensim-vc.pickle", mode="rb") as f:
		vec = pickle.load(f)
	print(cos_sim(vec["United_States"], vec["U.S."]))

