from graphviz import Digraph
from n_41 import cabocha_text2chunks


def visualize_cabocha(chunk_list, output_path):
	"""Chunkのリストを受け取り、係り受け木を出力する
	Args:
		chunk (list of Chunks): Chunkのリストとしての文
		output_path: outputのパス
	"""
	dot = Digraph()
	for idx, chunk in enumerate(chunk_list):
		if chunk.dst != -1:
			node = ''.join([morph.surface for morph in chunk.morphs if morph.pos != '記号'])
			dot.node(f"node{idx}", node)

	for idx in range(len(chunk_list)):
		if chunk_list[idx].dst != -1:
			dot.edge(f"node{idx}", f"node{chunk_list[idx].dst}")
	dot.render(output_path, view=True)

if __name__ == "__main__":
	processed = cabocha_text2chunks("./ai.ja.txt.parsed")
	visualize_cabocha(processed[1], "test.gv")
