from n_30 import load_parsed_text
from n_35 import get_word_freq
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Hiragino Maru Gothic Pro"]

morph_list = load_parsed_text("neko.txt.mecab")
freq_dic = get_word_freq(morph_list)

def plot_top_nfreq(freq_dic, n=10):
	"""単語基本形と頻度のタプルのリストを受け取り、上位n語とその出現頻度をグラフで表示する関数

	Args:
		freq_dic (list of tuple):　単語基本形と頻度のタプルのリスト
		n (int, optional): 上位何単語を出すかの変数 Defaults to 10.
	"""
	x = [freq_dic[:n][i][0] for i in range(n)]
	y = [freq_dic[:n][i][1] for i in range(n)]
	plt.bar(x, y)
	plt.title(f"単語頻度上位{n}位")
	plt.xlabel("単語")
	plt.ylabel("頻度（回）")
	plt.show()


if __name__ == "__main__":
	plot_top_nfreq(freq_dic)

