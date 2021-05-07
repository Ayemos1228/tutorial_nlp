from n_41 import cabocha_text2chunks
from n_43 import chunk_contains_pos
from n_45 import get_leftmost_verb, get_particles


def get_arguments(sentence, pred_chunk):
	"""文と動詞を含むchunkを受け取って、そのchunkにかかり助詞を含む文節のリスト返す関数

	Args:
		sentence : chunkのリストとしての文
		pred_chunk : 動詞を含むchunk
	Returns:
		arguments: 文節のリスト
	"""
	arguments = []
	src_chunks = [sentence[idx] for idx in pred_chunk.srcs]

	for chunk in src_chunks:
		if chunk_contains_pos(chunk, "助詞"):
			arguments.append("".join([morph.surface for morph in chunk.morphs if morph.pos != "記号"]))
	return arguments


def extract_case_frame(parsed, output_path):
	"""cabochaでパースされた文章を受け取って、日本語の述語格フレーム情報を出力。
	Args:
		parsed : cabochaでパースされた文章
		output_path: outputのパス
	Format:
		動詞	助詞1 助詞2 助詞3 ... 項1 項2 項3 ...
	"""
	frames = []
	predicate = ""
	particles = []

	for sentence in parsed:
		for chunk in sentence:
			if chunk_contains_pos(chunk, "動詞"):
				predicate = get_leftmost_verb(chunk)
				particles = get_particles(sentence, chunk)
				arguments = get_arguments(sentence, chunk)
				frames.append(predicate + "\t" + " ".join(particles) + " " + " ".join(arguments) +  "\n")

	# 書き込み
	with open(output_path, "w") as f:
		f.writelines(frames)


if __name__ == "__main__":
	parsed = cabocha_text2chunks("./ai.ja.txt.parsed")
	extract_case_frame(parsed, "./n_46_ans.txt")
