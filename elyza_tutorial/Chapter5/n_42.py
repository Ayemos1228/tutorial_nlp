from n_41 import Chunk, Morph, cabocha_text2chunks

if __name__ == '__main__':
	processed = cabocha_text2chunks("./ai.ja.txt.parsed")
	for sentence in processed[:2]:
		for chunk in sentence:
			print("".join([morph.surface  for morph in chunk.morphs if morph.pos != "記号"]), "".join([morph.surface for morph  in sentence[chunk.dst].morphs if morph.pos != "記号"]) if chunk.dst > 0 else "NONE", sep="\t")


