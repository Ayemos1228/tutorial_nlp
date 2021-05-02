import os
import sys
sys.path.append('/root/userspace/private/dl_nlp/chap5/')
import pickle

import numpy as np
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F

embedding_size = 300
n_filters = 100
latent_c_size = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VOCAB_PATH = '/root/userspace/private/dl_nlp/chap5/vocab.dump'

BASE_PATH = '/root/userspace/private/private/Assignments/{}/Chap 5/'

DISCRIMINATOR_PATH = '/root/userspace/private/dl_nlp/chap5/admin/model.pth'

def load_data(path):
    data = []
    for i, line in enumerate(open(path, encoding='utf-8')):
        words = line.strip().split()
        data.append(words)
    return data

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

def scoring(submission_path):
    # Load submission
    sens_submission = load_data(submission_path)

    test_submission = [vocab.sentence_to_ids(sen) for sen in sens_submission]
    test_submission = [torch.tensor(x, dtype=torch.long, device=device).unsqueeze(0) for x in test_submission]

    y_true = np.ones(500, dtype=np.int32).tolist() + np.zeros(500, dtype=np.int32).tolist()
    y_pred = []
    for x in test_submission:
        y_pred_ = torch.argmax(D.forward(x).squeeze()).item()
        y_pred.append(y_pred_)

    return accuracy_score(y_true, y_pred)


if __name__ == '__main__':
    # Get students list
    students = os.listdir('/root/userspace/private/private/Assignments')

    # Load pretrained model
    vocab = pickle.load(open(VOCAB_PATH, 'rb'))
    vocab_size = len(vocab.id2word)

    D = Discriminator(vocab_size, embedding_size, n_filters, latent_c_size).to(device)
    D.load_state_dict(torch.load(DISCRIMINATOR_PATH))

    for student in students:
        try:
            file_name = os.listdir(BASE_PATH.format(student))[0]
            submission_path = BASE_PATH.format(student) + file_name
            score = scoring(submission_path)
            print('{}: {}'.format(student, score))
        except Exception as e:
            print('{}: {}'.format(student, e))
