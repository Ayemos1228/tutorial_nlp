from n_30 import load_parsed_text
from n_35 import get_word_freq
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Hiragino Maru Gothic Pro"]

morph_list = load_parsed_text("neko.txt.mecab")
freq_dic = get_word_freq(morph_list)

def plot_freq_hist(freq_dic):
	"""単語基本形と頻度のタプルのリストを受け取り、単語の出現頻度のヒストグラムを描く関数

	Args:
		freq_dic (list of tuple):　単語基本形と頻度のタプルのリスト
	"""
	data = [word[1] for word in freq_dic]
	plt.hist(data, bins=100)
	plt.title("単語の出現頻度分布")
	plt.xlabel("出現頻度（回）")
	plt.ylabel("種類")
	plt.show()


if __name__ == "__main__":
	plot_freq_hist(freq_dic)

