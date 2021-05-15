import pickle
import gensim

model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
with open("./w2v.pickle", "wb") as f:
	pickle.dump(model, f)
