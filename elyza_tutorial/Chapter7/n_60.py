import gensim
import pickle


# vec = gensim.models.KeyedVectors.load_word2vec_format("./GoogleNews-vectors-negative300.bin", binary=True)
# with open("gensim-vc.pickle", mode="wb") as f:
# 	pickle.dump(vec,f)

with open("gensim-vc.pickle", mode="rb") as f:
	vec = pickle.load(f)
print(vec["United_States"])

