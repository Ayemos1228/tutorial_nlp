from n_30 import load_parsed_text
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Hiragino Maru Gothic Pro"]

morph_list = load_parsed_text("neko.txt.mecab")

def get_nmost_cooc_with_cat(parsed_sentences, n=10):
	"""形態素辞書のリストから「猫」と同じ文で観察された頻度が高い上位n語の頻度をプロットする関数

	Args:
		parsed_sentences (list of list of dic): 形態素辞書のリストとして表現された文のリスト
		n (int, optional): 上位n単語をプロット　default = 10
	"""
	freq_dic = {}
	for sentence in parsed_sentences:
		if "猫" in [x["base"] for x in sentence]:
			for word in sentence:
				if word["pos"] != "記号":
					freq_dic[word["base"]] = freq_dic.get(word["base"], 0) + 1
	del freq_dic["猫"]
	freq_dic = sorted(freq_dic.items(), key=lambda x:x[1], reverse=True)
	x = [freq_dic[:n][i][0] for i in range(n)]
	y = [freq_dic[:n][i][1] for i in range(n)]
	plt.bar(x, y)
	plt.title(f"「猫」との共起頻度上位{n}位")
	plt.xlabel("単語")
	plt.ylabel("共起頻度（回）")
	plt.show()


if __name__ == "__main__":
	print(get_nmost_cooc_with_cat(morph_list))


