from n_41 import cabocha_text2chunks
from n_43 import chunk_contains_pos
from n_45 import get_leftmost_verb, get_particles
from n_46 import get_arguments

def chunk_contains_nominal_verb(chunk):
	"""chunkが「サ変接続動詞＋を」の文節かどうかを判断する関数
	Args:
		chunk : 文節
	"""
	flag = 0
	if len(chunk.morphs) == 2:
		if chunk.morphs[0].pos == "名詞" and chunk.morphs[0].pos1 == "サ変接続":
			if chunk.morphs[1].pos == "助詞" and chunk.morphs[1].base == "を":
				flag = 1
	return flag

def get_particles_wo_vnouns(sentence, pred_chunk):
	"""文と動詞を含むchunkを受け取って、そのchunkにかかる助詞のリストを返す関数
	ただし、「サ変接続名詞＋を」の形の「を」は含めない。

	Args:
		sentence : chunkのリストとしての文
		pred_chunk : 動詞を含むchunk
	Returns:
		particles: 助詞のリスト（辞書順、重複なし）
	"""
	particles = set()
	src_chunks = [sentence[idx] for idx in pred_chunk.srcs]

	for chunk in src_chunks:
		if len(chunk.morphs) == 2:
				if chunk.morphs[0].pos == "名詞" and chunk.morphs[0].pos1 == "サ変接続":
					if chunk.morphs[1].pos == "助詞" and chunk.morphs[1].base == "を":
						continue
		for morph in chunk.morphs:
			if morph.pos == "助詞":
				particles.add(morph.surface)

	particles = sorted(list(particles))
	return particles


def get_nominal_verb(sentence, verb_chunk):
	"""文と動詞を含むchunkを受け取って、そのchunkにかかり「サ変接続名詞+を」で構成される文節を返す関数

	Args:
		sentence : chunkのリストとしての文
		pred_chunk : 動詞を含むchunk
	Returns:
		nverb (string): 条件を満たす文節のリスト
	"""
	nverb = ""
	src_chunks = [sentence[idx] for idx in verb_chunk.srcs]

	for chunk in src_chunks:
		if chunk_contains_nominal_verb(chunk):
			nverb = "".join([morph.surface for morph in chunk.morphs])

	return nverb


def extract_nominal_verbs(parsed, output_path):
	"""cabochaでパースされた文章を受け取って、日本語の述語格フレーム情報を出力。
	ただし、「サ変接続名詞＋を（助詞）」で構成される文節が動詞にかかる場合のみを考える。
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
				nominal_verb = get_nominal_verb(sentence, chunk)
				if nominal_verb != "":
					predicate = nominal_verb + get_leftmost_verb(chunk)
					particles = get_particles_wo_vnouns(sentence, chunk)
					arguments = get_arguments(sentence, chunk)
					arguments.remove(nominal_verb)
					if len(arguments) == 0:
						frames.append(predicate + "\n")
					else:
						frames.append(predicate + "\t" + " ".join(particles) + " " + " ".join(arguments) +  "\n")

	# 書き込み
	with open(output_path, "w") as f:
		f.writelines(frames)


if __name__ == "__main__":
	parsed = cabocha_text2chunks("./ai.ja.txt.parsed")
	extract_nominal_verbs(parsed, "./n_47_ans.txt")
