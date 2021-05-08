import pandas as pd
import numpy as np
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


def text_preprocess(sentence):
    """文に大文字を小文字になおし、数字はゼロにする処理を加える。
    Args:
            sentence (string): テクスト
    Returns:
            sentence (string): 下処理後のテクスト
    """
    sentence = sentence.lower()
    sentence = re.sub("[0-9]", "0", sentence)
    # 全角数字が変換されない（データになかったらごめん）かつ'123'みたいなのも'000'じゃなくて'0'に置換したほうがいいかも？精度変わらなかったらすみません
    return sentence


def get_stop_words(text, n=10):
    """テクストを受け取り、stop_wordとして取り除くべき頻度上位n語のリストを返す。

    Args:
            text (iterable of string): テクスト
            n (int, optional): ストップワードとして上位何単語を取り除くか
    Returns:
            stop_words (list): stop_wordのリスト
    """
    freq_dic = {}
    for sentence in text:
        sentence = sentence.split(" ")
        for word in sentence:
            freq_dic[word] = freq_dic.get(word, 0) + 1

    freq_dic = sorted(freq_dic.items(), key=lambda x: x[1], reverse=True)
    stop_words = [freq_dic[:n][i][0] for i in range(n)]

    return stop_words


if __name__ == "__main__":
    train_X = pd.read_csv("train.txt", sep="\t")["TITLE"]
    val_X = pd.read_csv("val.txt", sep="\t")["TITLE"]
    test_X = pd.read_csv("test.txt", sep="\t")["TITLE"]

    # preprocessing
    train_X = train_X.map(lambda x: text_preprocess(x))
    val_X = val_X.map(lambda x: text_preprocess(x))
    test_X = test_X.map(lambda x: text_preprocess(x))

    # stop_words
    stop_words = get_stop_words(train_X, 100)

    # vectorization
    # vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 2))
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    vectorizer.fit(train_X)

    train_X_transformed = vectorizer.transform(train_X).toarray()
    val_X_transformed = vectorizer.transform(val_X).toarray()
    test_X_transformed = vectorizer.transform(test_X).toarray()
    # to_dataframe
    train_df = pd.DataFrame(train_X_transformed, columns=vectorizer.get_feature_names())
    val_df = pd.DataFrame(val_X_transformed, columns=vectorizer.get_feature_names())
    test_df = pd.DataFrame(test_X_transformed, columns=vectorizer.get_feature_names())

    # write to files
    train_df.to_csv("train.feature.txt", sep="\t", index=False)
    val_df.to_csv("valid.feature.txt", sep="\t", index=False)
    test_df.to_csv("test.feature.txt", sep="\t", index=False)

    # pickle vectorizer
    with open("tf-idfvectorizer.pickle", mode="wb") as f:
        pickle.dump(vectorizer, f)
