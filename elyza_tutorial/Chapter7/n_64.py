import gensim
import pickle
import numpy as np

with open("gensim-vc.pickle", mode="rb") as f:
	vec = pickle.load(f)

with open("analogy.txt", mode="r") as f:
	with open("analogy_ans.txt", mode="w") as g:
		lines = f.readlines()
		for line in lines:
			# categoryの変わり目
			if line[0] == ":":
				g.write(line)
				continue
			# それ以外
			words = line.rstrip("\n").split(" ")
			target_vec = vec[words[1]] - vec[words[0]] + vec[words[2]]
			ans = vec.similar_by_vector(target_vec, topn=1)
			words.extend([ans[0][0], str(ans[0][1])])
			g.write(" ".join(words) + "\n")

