import os
import csv
import torch
import torchvision


ANSWER_PATH = 'answer.csv'

DIR = os.path.dirname(__file__)
data_path = os.path.join(DIR, '../data/fashion-mnist')
answer_path = os.path.join(DIR, ANSWER_PATH)


test_data = torchvision.datasets.FashionMNIST(
    data_path,
    transform=torchvision.transforms.ToTensor(),
    train=False,
    download=True)

test_data_loader = torch.utils.data.DataLoader(
    dataset=test_data,
    batch_size=100,
    shuffle=False)

y_true = []

for _, t in test_data_loader:
    y_true += t.tolist()

with open(answer_path, 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(y_true)
