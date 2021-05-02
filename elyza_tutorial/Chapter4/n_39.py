from n_30 import load_parsed_text
from n_35 import get_word_freq
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Hiragino Maru Gothic Pro"]

morph_list = load_parsed_text("neko.txt.mecab")
freq_dic = get_word_freq(morph_list)

def plot_zipf(freq_dic):
	"""単語基本形と頻度のタプルのリストを受け取り、
	横軸：出現頻度順位
	縦軸：出現頻度
	とする両対数グラフを描く関数　

	Args:
		freq_dic (list of tuple):　単語基本形と頻度のタプルのリスト
	"""
	x = [i + 1 for i in range(len(freq_dic))]
	y = [word[1] for word in freq_dic]

	plt.plot(x, y)
	plt.title("Zipfの法則")
	plt.xlabel("出現頻度順位（位）")
	plt.ylabel("出現頻度（回）")
	plt.xscale("log")
	plt.yscale("log")
	plt.show()


if __name__ == "__main__":
	plot_zipf(freq_dic)
