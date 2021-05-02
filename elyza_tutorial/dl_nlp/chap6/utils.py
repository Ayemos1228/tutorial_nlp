import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Vocab(object):
    def __init__(self, word2id={}):
        """
        word2id: 単語(str)をインデックス(int)に変換する辞書
        id2word: インデックス(int)を単語(str)に変換する辞書
        """
        self.word2id = dict(word2id)
        self.id2word = {v: k for k, v in self.word2id.items()}    
        
    def build_vocab(self, sentences, min_count=1):
        # 各単語の出現回数の辞書を作成する
        word_counter = {}
        for sentence in sentences:
            for word in sentence:
                word_counter[word] = word_counter.get(word, 0) + 1

        # min_count回以上出現する単語のみ語彙に加える
        for word, count in sorted(word_counter.items(), key=lambda x: -x[1]):
            if count < min_count:
                break
            _id = len(self.word2id)
            self.word2id.setdefault(word, _id)
            self.id2word[_id] = word 


class TextGenerator(nn.Module):
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim,
                 sequence_length):
        """
        :param num_emb: int, 語彙の総数
        :param batch_size: int, ミニバッチのサイズ
        :param emb_dim: int, 埋め込みベクトルの次元数
        :param hidden_dim: int, 隠れ状態ベクトルの次元数
        :param sequence_length: int, 入出力系列の長さ
        """
        super(TextGenerator, self).__init__()
        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        
        # 埋め込み層
        self.embedding = nn.Embedding(self.num_emb, self.emb_dim)
        # LSTM
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_dim, 1, batch_first=True)
        # 全結合層
        self.linear = nn.Linear(self.hidden_dim, self.num_emb)

    def forward(self, inputs):
        """
        ターゲット系列を全時刻での入力として出力を計算
        :param inputs: torch.Tensor, (batch_size, sequence_length)
        :return outputs: torch.Tensor, (batch_size*sequence_length, num_emb)
        """
        N = inputs.size(0) # batch_size
        embed = self.embedding(inputs) # (batch_size, sequence_length, emb_dim)
        h0, c0 = self.init_hidden(N) # 隠れ状態ベクトルの初期化
        h = (h0, c0)
        self.lstm.flatten_parameters()
        hidden, h = self.lstm(embed, h) # hidden:(batch_size, sequence_length, hidden_dim)
        lin = self.linear(hidden) # (batch_size, sequence_length, num_emb)
        outputs = F.log_softmax(lin, dim=-1) # (batch_size, sequence_length, num_emb)
        outputs = outputs.view(-1, self.num_emb) # (batch_size * sequence_length, num_emb)
        return outputs

    def step(self, x, h, c):
        """
        時刻をtからt+1に1つだけ進めます
        :param x: torch.Tensor, 時刻tの出力かつ時刻t+1の入力
        :param h, c: torch.Tensor, 時刻tの隠れ状態ベクトル
        :return pred: torch.Tensor, 時刻t+1の出力
        :return h, c: torch.Tensor, 時刻t+1の隠れ状態ベクトル
        """
        embed = self.embedding(x) # embed:(batch_size, 1, emb_dim)
        self.lstm.flatten_parameters()
        y, (h, c) = self.lstm(embed, (h, c)) # y:(batch_size, 1, hidden_dim)
        pred = F.softmax(self.linear(y), dim=-1) # (batch_size, 1, num_emb)
        return pred, h, c

    def sample(self, x=None):
        """
        Generaterでサンプリングするメソッド
        :param x: None or torch.Tensor
        :param output: torch.Tensor, (batch_size, sequence_length)
        """
        flag = False # 時刻0から始める(True)か否か(False)
        if x is None:
            flag = True
        if flag:
            x = torch.empty(self.batch_size, 1).fill_(1).long().to(device) # BOS == 1
        h, c = self.init_hidden(self.batch_size)

        samples = []
        if flag:
            for i in range(self.sequence_length):
                output, h, c = self.step(x, h, c) # output:(batch_size, 1, num_emb)
                output = output.squeeze(1) # (batch_size, num_emb)
                x = output.multinomial(1) # (batch_size, 1), 次の時刻の入力を多項分布からサンプリング
                samples.append(x)
        else:
            given_len = x.size(1)
            lis = x.chunk(x.size(1), dim=1) # sequence_length方向に分割
            for i in range(given_len):
                output, h, c = self.step(lis[i], h, c)
                samples.append(lis[i])
            output = output.squeeze(1)
            x = output.multinomial(1)
            for i in range(given_len, self.sequence_length):
                samples.append(x)
                output, h, c = self.step(x, h, c)
                output = output.squeeze(1)
                x = output.multinomial(1)
        output = torch.cat(samples, dim=1)
        return output

    def init_hidden(self, N):
        """
        LSTMの隠れ状態ベクトルを初期化します。
        :param N: int, ミニバッチのサイズ
        """
        h0 = torch.zeros(1, N, self.hidden_dim).to(device)
        c0 = torch.zeros(1, N, self.hidden_dim).to(device)
        return h0, c0


class TargetLSTM(nn.Module):
    def __init__(self, num_emb, emb_dim, hidden_dim,
                 sequence_length, start_token):
        super(TargetLSTM, self).__init__()
        self.num_emb = num_emb
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token = start_token

        self.embedding = nn.Embedding(self.num_emb, self.emb_dim)
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_dim, 1, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, self.num_emb)
        self.init_params()
        
    def forward(self, inputs):
        N = inputs.size(0)
        embed = self.embedding(inputs)
        h0, c0 = self.init_hidden(N)
        h = (h0, c0)
        self.lstm.flatten_parameters()
        hidden, h = self.lstm(embed, h)
        outputs = F.log_softmax(self.linear(hidden), dim=-1)
        return outputs.view(-1, self.num_emb)

    def step(self, x, h, c):
        x = self.embedding(x)
        self.lstm.flatten_parameters()
        y, (h, c) = self.lstm(x, (h, c))
        pred = F.softmax(self.linear(y), dim=-1)
        return pred, h, c

    def sample(self, N):
        samples = []
        x = torch.tensor([self.start_token] * N, dtype=torch.long).unsqueeze(1).to(device)
        h, c = self.init_hidden(N)
        for i in range(self.sequence_length):
            pred, h, c = self.step(x, h, c)
            pred = pred.squeeze(1)
            x = pred.multinomial(1)
            samples.append(x)
        samples = torch.cat(samples, dim=1)
        return samples

    def init_params(self):
        for param in self.parameters():
            param.data.normal_(mean=0, std=1)

    def init_hidden(self, N):
        # (num_layers*direction, batch_size, hidden_dim)
        h0 = torch.zeros(1, N, self.hidden_dim).to(device)
        c0 = torch.zeros(1, N, self.hidden_dim).to(device)
        return h0, c0
