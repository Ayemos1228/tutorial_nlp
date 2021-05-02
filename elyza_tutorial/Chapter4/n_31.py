from n_30 import load_parsed_text

morph_list = load_parsed_text("neko.txt.mecab")

def extract_sform(parsed_sentences):
	"""形態素辞書のリストから動詞の表層形を抽出する関数

	Args:
		parsed_sentences (list of list of dic): 形態素辞書のリストとして表現された文のリスト
	Return:
		sform_set (set) : 動詞の表層形の集合
	"""
	sform_set = set()
	for sentence in parsed_sentences:
		for word in sentence:
			if word["pos"] == "動詞":
				sform_set.add(word["surface"])
	return sform_set


if __name__ == "__main__":
	print(extract_sform(morph_list))
