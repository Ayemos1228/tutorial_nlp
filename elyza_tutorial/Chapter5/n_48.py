from n_41 import cabocha_text2chunks
from n_43 import chunk_contains_pos
from n_45 import get_leftmost_verb, get_particles
from n_46 import get_arguments



def extract_path_from_nouns(parsed, output_path):
	"""cabochaでパースされた文章を受け取って、
	全ての名詞を含む文節に対してその文節から構文木の根に至るパスを抽出する関数。
	Args:
		parsed : cabochaでパースされた文章
		output_path: outputのパス
	Format:
		”AIに関する -> 最初の -> 会議で -> 作り出した”
	"""
	paths = []
	for idx, sentence in enumerate(parsed):
		paths.append(f"---sentence no.{idx}---" + "\n")
		for chunk in sentence:
			if chunk_contains_pos(chunk, "名詞"):
				path = []
				chunk_pres = chunk
				chunk_surface = ''.join([morph.surface for morph in chunk_pres.morphs if morph.pos != '記号'])
				path.append(chunk_surface)
				while chunk_pres.dst != -1: #  係り先の文節がなくなるまで
					chunk_pres = sentence[chunk_pres.dst]
					chunk_surface = ''.join([morph.surface for morph in chunk_pres.morphs if morph.pos != '記号'])
					path.append(chunk_surface)
				paths.append(" -> ".join(path) + "\n")

	with open(output_path, "w") as f:
		f.writelines(paths)


if __name__ == "__main__":
	processed = cabocha_text2chunks("./ai.ja.txt.parsed")
	extract_path_from_nouns(processed, "n_48_ans.txt")
