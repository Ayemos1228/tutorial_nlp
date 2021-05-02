from n_30 import load_parsed_text

morph_list = load_parsed_text("neko.txt.mecab")

def get_NO_connected_np(parsed_sentences):
	"""形態素辞書のリストから「AのB」の形の名詞句を取り出す関数

	Args:
		parsed_sentences (list of list of dic): 形態素辞書のリストとして表現された文のリスト
	Return:
		connected_np (set) : 「AのB」の形の名詞句の集合
	"""
	connected_np = set()
	for sentence in parsed_sentences:
		for no_idx in range(1, len(sentence) - 1): # i: 「の」が現れうるindex
			if (sentence[no_idx - 1]["pos"] == "名詞" and \
				sentence[no_idx]["base"] == "の" and \
				sentence[no_idx + 1]["pos"] == "名詞"):
				connected_np.add(sentence[no_idx - 1]["surface"] + "の" + sentence[no_idx + 1]["surface"])
	return connected_np

if __name__ == "__main__":
	print(get_NO_connected_np(morph_list))
