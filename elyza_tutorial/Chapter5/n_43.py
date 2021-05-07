from n_41 import Chunk, Morph, cabocha_text2chunks


def chunk_contains_pos(chunk, pos):
	"""文節があるposを含むかを判定する関数
	Args:
		chunk: 文節
		pos: 判定したい品詞
	Returns: posを含めば1, 含まなければ0
	"""
	return (pos in [morph.pos for morph in chunk.morphs])


if __name__ == '__main__':
	processed = cabocha_text2chunks("./ai.ja.txt.parsed")
	for sentence in processed[:2]:
		for chunk in sentence:
			if chunk.dst > 0 and chunk_contains_pos(chunk, "名詞") and chunk_contains_pos(sentence[chunk.dst], "動詞"):
					print("".join([morph.surface for morph in chunk.morphs if morph.pos != "記号"]), "".join([morph.surface for morph  in sentence[chunk.dst].morphs if morph.pos != "記号"]), sep="\t")


