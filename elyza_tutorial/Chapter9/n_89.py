import pickle

with open("./w2v.pickle", "rb") as f:
	vec = pickle.load(f)

if "word" in vec.key_to_index.keys():
	print("ok")
