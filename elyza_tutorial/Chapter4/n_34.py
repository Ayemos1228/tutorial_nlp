from n_30 import load_parsed_text

morph_list = load_parsed_text("neko.txt.mecab")

def get_connected_np(parsed_sentences):
	"""形態素辞書のリストから名詞の連接を最長一致で抽出する関数
	Args:
		parsed_sentences (list of list of dic): 形態素辞書のリストとして表現された文のリスト
	Return:
		connected_np (set) : 連接された名詞句の集合
	"""
	connected_np = set()
	for sentence in parsed_sentences:
		connected_word = ""
		word_len = 0
		for word in sentence:
			if word["pos"] == "名詞":
				connected_word += word["surface"]
				word_len += 1
			elif word_len >= 2:
				connected_np.add(connected_word)
				word_len = 0
				connected_word = ""
			else:
				word_len = 0
				connected_word = ""
		if word_len >= 2: #文の最後までが連接名詞句である場合を考慮
			connected_np.add(connected_word)
	return connected_np

if __name__ == "__main__":
	print(get_connected_np(morph_list))
