from n_30 import load_parsed_text

morph_list = load_parsed_text("neko.txt.mecab")

def get_word_freq(parsed_sentences):
	"""形態素辞書のリストから単語の出現頻度を抽出して頻度順に並べる関数

	Args:
		parsed_sentences (list of list of dic): 形態素辞書のリストとして表現された文のリスト
	Return:
		word_freq (list of tuple) : [(単語の基本形 : 頻度), ...] （頻度の降順）
	"""
	word_freq = {}
	for sentence in parsed_sentences:
		for word in sentence:
			if word["pos"] != "記号":
				word_freq[word["base"]] = word_freq.get(word["base"], 0) + 1
	word_freq = sorted(word_freq.items(), key=lambda x:x[1], reverse=True)
	return word_freq


if __name__ == "__main__":
	print(get_word_freq(morph_list)[:20])

