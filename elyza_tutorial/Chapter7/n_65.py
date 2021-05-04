import gensim
import pickle
import numpy as np


sem_result = []
syn_result = []
is_semdata = True

with open("analogy_ans.txt", mode="r") as f:
	lines = f.readlines()
	for line in lines:
		# categoryの変わり目
		if line[0] == ":":
			if "gram" not in line:
				continue
			is_semdata = False
			continue
		#　それ以外
		words = line.split(" ")
		if is_semdata:
			sem_result.append(words[3] == words[4])
		else:
			syn_result.append(words[3] == words[4])

print(f"accuracy on semantic dataset: {sum(sem_result) / len(sem_result)}")
print()
print(f"accuracy on syntactic dataset: {sum(syn_result) / len(syn_result)}")
