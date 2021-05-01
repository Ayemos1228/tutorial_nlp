import MeCab

tagger = MeCab.Tagger("-Ochasen")
parsed = tagger.parse("neko.txt")
print(parsed)
