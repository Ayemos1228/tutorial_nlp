class Morph:
	def __init__(self, surface, base, pos, pos1):
		self.surface = surface
		self.base = base
		self.pos = pos
		self.pos1 = pos1

	def info(self):
		print(f"surface: {self.surface}, base: {self.base}, pos: {self.pos}, pos1: {self.pos1}")

class Chunk:
	def __init__(self, morphs, dst):
		self.morphs = morphs # 形態素のリスト
		self.dst = dst # 係り先文節のインデックス
		self.srcs = [] # 係り元文節インデックスのリスト

def cabocha_text2chunks(path):
	"""cabochaでparseされた結果を読み込んで、各文をChunk(文節）のリストとして返す関数
	Args:
		path (str): textへのpath
	Return:
		sentences (list of list of Morph): Chunkのリストとして表された各文のリスト
	"""
	sentences = []
	sentence = []
	chunk = []
	dst_nums = []
	with open(path, "r") as f:
		for line in f:
			# sentence, chunk, dst_numはここで定義すれば最後に初期化せずに済む
			if line.startswith("*"): # * から始まるところはスキップ
				if len(chunk) != 0:
					sentence.append(chunk)
					dst_nums.append(dst_num)
					chunk = []
				dst_num = int(line.split()[2].rstrip("D"))
			elif line != "EOS\n": # 文中
				split_line = line.rstrip("\n").split("\t")
				morph_info = split_line[1].split(",")
				morph = Morph(
					split_line[0],
					morph_info[6],
					morph_info[0],
					morph_info[1])
				chunk.append(morph)
			else: # 文の終わり
				if len(chunk) != 0:
					sentence.append(chunk)
					dst_nums.append(dst_num)
				if len(sentence) != 0:
					chunk_list = []
					for now, morphs_ in enumerate(sentence):
						chunk_obj = Chunk(morphs_, dst_nums[now])
						# dst_numsで値がdstになってるidxの値をchunk.srcsに入れる
						for i in range(len(dst_nums)):
							if dst_nums[i] == now:
								chunk_obj.srcs.append(i)
						chunk_list.append(chunk_obj)
					sentences.append(chunk_list)
				sentence = []
				chunk = []
				dst_nums = []
	return sentences


if __name__ == '__main__':
	ans = cabocha_text2chunks("./ai.ja.txt.parsed")
	for chunk in ans[1]:
		print([morph.surface for morph in chunk.morphs], chunk.dst, chunk.srcs)
