import MeCab
NEKO_PATH = "./neko.txt"

def load_parsed_text(path):
	"""parseされた結果を読み込んで一文ずつリストにする関数
	Args:
		path (str): textへのpath
	Return:
		morph_list (list of list of dic): 各形態素の情報が入った辞書のリスト
	"""
	morph_list = []
	sentence = []
	with open(path, "r") as f:
		for line in f:
			if line != "EOS\n": # 文中
				split_line = line.split("\t")
				morph_info = split_line[1].split(",")
				morph_dic = {
					"surface": split_line[0],
					"base": morph_info[6],
					"pos": morph_info[0],
					"pos1": morph_info[1]}
				sentence.append(morph_dic)
			else: # 文の終わり
				if len(sentence) != 0:
					morph_list.append(sentence)
					sentence = []
	return morph_list


if	__name__ == '__main__':
	tagger = MeCab.Tagger()
	with open(NEKO_PATH, "r") as f:
		with open("neko.txt.mecab", "w") as g:
			for line in f:
				g.write(tagger.parse(line))
	print(load_parsed_text("neko.txt.mecab"))
