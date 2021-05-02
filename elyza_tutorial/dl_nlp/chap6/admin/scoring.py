import torch
import torch.nn as nn
import sys
sys.path.append("..")
from utils import TargetLSTM

def load_data(path):
    with open(path, "r") as f:
        data = []
        for line in f:
            line = list(map(int, line.strip().split(",")))
            data.append(line)
    return data

def evaluate():
    generated_data = load_data("../data/submission.csv")
    generated_num = len(generated_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = 5000
    G_EMB_DIM = 32 # 埋め込みベクトルの次元数
    G_HIDDEN_DIM = 32 # LSTMの隠れ状態ベクトルの次元数
    G_SEQ_LENGTH = 20 # 系列の長さ
    BOS = 0

    target_lstm = TargetLSTM(vocab_size, G_EMB_DIM, G_HIDDEN_DIM, G_SEQ_LENGTH, BOS).to(device)
    target_lstm.load_state_dict(torch.load("./target_lstm_params.pth", map_location="cpu"))

    pointer = 0
    batch_size = 64

    criterion = nn.NLLLoss(size_average=False)
    total_loss = 0
    total_tokens = 0

    while pointer < generated_num:
        data = generated_data[pointer:pointer + batch_size]
        data = torch.tensor(data, dtype=torch.long, device=device)
        start_tokens = torch.empty(data.size(0), 1).fill_(BOS).long().to(device)
        source = torch.cat([start_tokens, data[:, :-1]], dim=1)
        pred = target_lstm(source)
        loss = criterion(pred, data.view(-1,))

        total_loss += loss.item()
        total_tokens += data.size(0) * data.size(1)
        pointer += batch_size

    loss = total_loss / total_tokens

    return loss


if __name__ == '__main__':
    print(evaluate())
