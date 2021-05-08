from n_41 import cabocha_text2chunks
from n_43 import chunk_contains_pos
from n_45 import get_leftmost_verb, get_particles
from n_46 import get_arguments
from itertools import combinations


def replace_nouns_with_XY(chunk, char):
	"""chunkのうち名詞の連続をcharに置き換える関数
	Args:
		chunk : 文節
		char :  置き換え文字
	Returns:
		replaced: 置き換え後の文節
	"""
	replaced = []
	morphs = [morph for morph in chunk.morphs if morph.pos != "記号"]
	i = 0
	while i < len(morphs):
		if morphs[i].pos == "名詞":
			while i < len(morphs) and morphs[i].pos == "名詞":
				i += 1
			replaced.append(char)
		else:
			replaced.append(morphs[i].surface)
			i += 1

	return "".join(replaced)


def extract_path_between_nouns(sentence, output_path):
	"""cabochaでパースされた文章を受け取って、
	文中の全ての名詞句のペアを結ぶ最短係り受けパスを抽出する関数。
	Args:
		parsed : cabochaでパースされた文章
		output_path: outputのパス
	"""

	n_num = []
	for idx, chunk in enumerate(sentence):
		if chunk_contains_pos(chunk, "名詞"):
			n_num.append(idx)
	for i, j in combinations(n_num, 2):
		path_from_i = []
		path_from_j = []
		while i != j:
			if i < j:
				path_from_i.append(i)
				i = sentence[i].dst
			else:
				path_from_j.append(j)
				j = sentence[i].dst

		if len(path_from_j) == 0:
			chunk_i = replace_nouns_with_XY(sentence[path_from_i[0]], "X")
			chunk_j =  replace_nouns_with_XY(sentence[i], "Y")
			words_between_iandj =  [''.join(morph.surface for morph in sentence[n].morphs) for n in path_from_i[1:]]
			print(chunk_i + " -> " + " ->".join(words_between_iandj) + " -> " + chunk_j)
		else:
			chunk_i = replace_nouns_with_XY(sentence[path_from_i[0]], "X")
			chunk_j =  replace_nouns_with_XY(sentence[path_from_j[0]], "Y")
			chunk_k = ''.join([morph.surface for morph in sentence[i].morphs if morph.pos != "記号"])
			words_itok = [chunk_i] + [''.join([morph.surface for morph in sentence[n].morphs if morph.pos != "記号"]) for n in path_from_i[1:]]
			words_jtok = [chunk_j] + [''.join([morph.surface for morph in sentence[n].morph if morph.pos != "記号"]) for n in path_from_j[1:]]
			print(' | '.join([' -> '.join(words_itok), ' -> '.join(words_jtok), chunk_k]))


if __name__ == "__main__":
	parsed = cabocha_text2chunks("./ai.ja.txt.parsed")
	extract_path_between_nouns(parsed[1], "n_49_ans.txt")
