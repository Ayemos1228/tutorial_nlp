import time
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

np.random.seed(34)
torch.manual_seed(34)

num_epochs = 3
batch_size = 64

embedding_size = 300
n_filters = 100
latent_c_size = 2

min_count = 1

PAD = 0
BOS = 1
EOS = 2
UNK = 3
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
BOS_TOKEN = '<S>'
EOS_TOKEN = '</S>'

TRAIN_X_PATH = '../data/styletransfer/train_x.txt'
TRAIN_Y_PATH = '../data/styletransfer/train_y.txt'
VALID_X_PATH = '../data/styletransfer/valid_x.txt'
VALID_Y_PATH = '../data/styletransfer/valid_y.txt'

VOCAB_PATH = '../vocab.dump'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(path, n_data=10e+10):
    data = []
    for i, line in enumerate(open(path, encoding='utf-8')):
        words = line.strip().split()
        data.append(words)
        if i + 1 >= n_data:
            break
    return data

class DataLoader:
    def __init__(self, data_x, data_y, batch_size, shuffle=True):
        """
        :param data_x: list, 文章 (単語IDのリスト) のリスト
        :param data_y: list, 属性ラベルのリスト
        :param batch_size: int, バッチサイズ
        :param shuffle: bool, サンプルの順番をシャッフルするか否か
        """
        self.data = list(zip(data_x, data_y))

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.start_index = 0

        self.reset()

    def reset(self):
        if self.shuffle:
            self.data = shuffle(self.data)
        self.start_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        # ポインタが最後まで到達したら初期化する
        if self.start_index >= len(self.data):
            self.reset()
            raise StopIteration()

        # バッチを取得
        batch_x, batch_c = zip(*self.data[self.start_index:self.start_index+self.batch_size])

        # 系列長で降順にソート
        batch = sorted(zip(batch_x, batch_c), key=lambda x: len(x[0]), reverse=True)
        batch_x, batch_c = zip(*batch)

        # 系列長を取得
        batch_x = [[BOS] + x + [EOS] for x in batch_x]
        batch_x_lens = [len(x) for x in batch_x]

        # <S>, </S>を付与 + 短い系列にパディング
        max_length = max(batch_x_lens)
        batch_x = [x + [PAD] * (max_length - len(x)) for x in batch_x]

        # tensorに変換
        batch_x = torch.tensor(batch_x, dtype=torch.long, device=device)
        batch_c = torch.tensor(batch_c, dtype=torch.long, device=device)

        # ポインタを更新する
        self.start_index += self.batch_size

        return batch_x, batch_c, batch_x_lens

sens_train_X = load_data(TRAIN_X_PATH)
sens_valid_X = load_data(VALID_X_PATH)

vocab = pickle.load(open(VOCAB_PATH, 'rb'))

train_X = [vocab.sentence_to_ids(sen) for sen in sens_train_X]
valid_X = [vocab.sentence_to_ids(sen) for sen in sens_valid_X]

train_Y = np.loadtxt(TRAIN_Y_PATH)
valid_Y = np.loadtxt(VALID_Y_PATH)

vocab_size = len(vocab.word2id)
print('語彙数:', vocab_size)
print('学習用データ数:', len(train_X))
print('検証用データ数:', len(valid_X))

dataloader_train = DataLoader(train_X, train_Y, batch_size)
dataloader_valid = DataLoader(valid_X, valid_Y, batch_size)

class Discriminator(nn.Module):
    def __init__(self, vocab_size, embedding_size, n_filters, latent_c_size):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.conv_1 = nn.Conv1d(embedding_size, n_filters, kernel_size=3)
        self.conv_2 = nn.Conv1d(embedding_size, n_filters, kernel_size=4)
        self.conv_3 = nn.Conv1d(embedding_size, n_filters, kernel_size=5)
        self.out = nn.Linear(n_filters*3, latent_c_size)

    def forward(self, x):
        """
        :param x: tensor, 単語id列のバッチ or 単語の確率分布列のバッチ, size=(バッチサイズ, 系列長) or (バッチサイズ, 系列長, 語彙数)
        :return x: tensor, size=(バッチサイズ, latent_c_size)
        """
        x_len = x.size(1)

        if x.dim() == 2: # xが単語IDのtensorの場合
            x = self.embedding(x)
        elif x.dim() == 3: # xが単語の確率分布のtensorの場合
            x = torch.matmul(x, self.embedding.weight)

        x = x.permute(0, 2, 1) # conv用に次元を入れ替え, size=(バッチサイズ, embedding_size, 系列長)

        x1 = torch.tanh(self.conv_1(x)) # フィルター1に対して畳み込み, size=(バッチサイズ, フィルター数, 系列長-2)
        x2 = torch.tanh(self.conv_2(x)) # フィルター1に対して畳み込み, size=(バッチサイズ, フィルター数, 系列長-3)
        x3 = torch.tanh(self.conv_3(x)) # フィルター1に対して畳み込み, size=(バッチサイズ, フィルター数, 系列長-4)

        x1 = F.max_pool1d(x1, x_len-2) # x1に対してpooling, size=(バッチサイズ, フィルター数, 1)
        x2 = F.max_pool1d(x2, x_len-3) # x2に対してpooling, size=(バッチサイズ, フィルター数, 1)
        x3 = F.max_pool1d(x3, x_len-4) # x3に対してpooling, size=(バッチサイズ, フィルター数, 1)

        x = torch.cat([x1, x2, x3], dim=1)[:, :, 0] # size=(バッチサイズ, フィルター数*3)
        x = self.out(x)

        return x # (バッチサイズ, latent_c_size)

D_args = {
    'vocab_size': vocab_size,
    'embedding_size': embedding_size,
    'n_filters': n_filters,
    'latent_c_size': latent_c_size
}

D = Discriminator(**D_args).to(device)

optimizer_D = optim.Adam(D.parameters())

def compute_loss_D(x, c, is_train):
    input_encoder = x[:, 1:-1] # [x_1, x_2, ..., x_T]
    c_pred = D.forward(input_encoder) # (バッチサイズ, latent_c_size)

    loss_D = F.cross_entropy(c_pred, c)

    if is_train:
        D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

    return loss_D

start_time = time.time()
for epoch in range(num_epochs):
    # Train
    D.train()
    loss_D_train = []
    for batch_x, batch_c, _ in tqdm(dataloader_train):
        # Discriminatorの学習
        loss_D = compute_loss_D(batch_x, batch_c, is_train=True)
        loss_D_train.append(loss_D.item())

    # Valid
    D.eval()
    loss_D_valid = []

    pred_valid = []
    gold_valid = []
    for batch_x, batch_c, _ in dataloader_valid:
        # Discriminatorの検証
        loss_D = compute_loss_D(batch_x, batch_c, is_train=False)
        loss_D_valid.append(loss_D.item())

        c_pred = D.forward(batch_x).argmax(dim=1).tolist()
        c_gold = batch_c.tolist()
        pred_valid.extend(c_pred)
        gold_valid.extend(c_gold)

    print('EPOCH: {}, Train Loss: {:.2f}, Valid Loss: {:.2f}, Accuracy: {:.2f}'.format(
        epoch + 1,
        np.mean(loss_D_train),
        np.mean(loss_D_valid),
        accuracy_score(gold_valid, pred_valid),
    ))

state_dict = D.state_dict()
torch.save(state_dict, './model.pth')
