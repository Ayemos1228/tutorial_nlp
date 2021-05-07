from n_41 import Chunk, Morph, cabocha_text2chunks
from n_43 import chunk_contains_pos


def get_leftmost_verb(chunk):
	"""動詞を含むチャンクを与えられて最も左の動詞を返す関数

	Args:
		chunk : 文節
	Returns:
		verb: 一番左の動詞
	"""
	for word in chunk.morphs:
		if word.pos == "動詞":
			verb = word.base
			break
	return verb


def get_particles(sentence, pred_chunk):
	"""文と動詞を含むchunkを受け取って、そのchunkにかかる助詞のリストを返す関数

	Args:
		sentence : chunkのリストとしての文
		pred_chunk : 動詞を含むchunk
	Returns:
		particles: 助詞のリスト（辞書順、重複なし）
	"""
	particles = set()
	src_chunks = [sentence[idx] for idx in pred_chunk.srcs]

	for chunk in src_chunks:
		for morph in chunk.morphs:
			if morph.pos == "助詞":
				particles.add(morph.surface)
	particles = sorted(list(particles))
	return particles


def extract_case_pattern(parsed, output_path):
	"""cabochaでパースされた文章を受け取って、日本語の述語格パターンを出力。
	Args:
		parsed : cabochaでパースされた文章
		output_path: outputのパス
	Format:
		動詞	助詞1 助詞2 助詞3 ...
	"""
	patterns = []
	predicate = ""
	particles = []

	for sentence in parsed:
		for chunk in sentence:
			if chunk_contains_pos(chunk, "動詞"):
				predicate = get_leftmost_verb(chunk)
				particles = get_particles(sentence, chunk)
				patterns.append(predicate + "\t" + " ".join(particles) + "\n")

	# 書き込み
	with open(output_path, "w") as f:
		f.writelines(patterns)



if __name__ == "__main__":
	parsed = cabocha_text2chunks("./ai.ja.txt.parsed")
	extract_case_pattern(parsed, "./n_45_ans.txt")

	# 確認
	# cat ./n_45_ans.txt | sort | uniq -c | sort -nr | head -n 10
	# cat ./n_45_ans.txt | grep "行う" | sort | uniq -c | sort -nr | head -n 10
	# cat ./n_45_ans.txt | grep "なる" | sort | uniq -c | sort -nr | head -n 10
	# cat ./n_45_ans.txt | grep "与える" | sort | uniq -c | sort -nr | head -n 10
