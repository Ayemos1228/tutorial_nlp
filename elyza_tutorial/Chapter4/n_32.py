from n_30 import load_parsed_text

morph_list = load_parsed_text("neko.txt.mecab")

def extract_nform(parsed_sentences):
	"""形態素辞書のリストから動詞の基本形を抽出する関数

	Args:
		parsed_sentences (list of list of dic): 形態素辞書のリストとして表現された文のリスト
	Return:
		nform_set (set) : 動詞の基本形の集合
	"""
	nform_set = set()
	for sentence in parsed_sentences:
		for word in sentence:
			if word["pos"] == "動詞":
				nform_set.add(word["base"])
	return nform_set


if __name__ == "__main__":
	print(extract_nform(morph_list))
