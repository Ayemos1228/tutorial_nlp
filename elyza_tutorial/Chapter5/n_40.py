class Morph:
	def __init__(self, surface, base, pos, pos1):
		self.surface = surface
		self.base = base
		self.pos = pos
		self.pos1 = pos1

	def info(self):
		print(f"surface: {self.surface}, base: {self.base}, pos: {self.pos}, pos1: {self.pos1}")

def load_cabocha_text(path):
	"""cabochaでparseされた結果を読み込んでMorphクラスに各形態素を登録して、
	各文をMorphのリストとして返す関数
	Args:
		path (str): textへのpath
	Return:
		sentences (list of list of Morph): Morphのリストとして表された各文のリスト
	"""
	sentences = []
	sentence = []
	with open(path, "r") as f:
		for line in f:
			if line.startswith("*"): # * から始まるところはスキップ
				continue
			elif line != "EOS\n": # 文中
				split_line = line.rstrip("\n").split("\t")
				morph_info = split_line[1].split(",")
				word = Morph(
					split_line[0],
					morph_info[6],
					morph_info[0],
					morph_info[1])
				sentence.append(word)
			else: # 文の終わり
				if len(sentence) != 0:
					sentences.append(sentence)
					sentence = []
	return sentences

if	__name__ == '__main__':
	cabo_text = load_cabocha_text("./ai.ja.txt.parsed")
	for morph in cabo_text[1]:
		morph.info()
