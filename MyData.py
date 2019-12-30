import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MyData(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        target = self.target[index]
        return [data, target]


def collate_fn(data):
    input, target = list(list(zip(*data))[0]), list(list(zip(*data))[1])
    return input, target


def get_loader(data, target, batch_size):
    dataset = MyData(data, target)
    dataloader = DataLoader(dataset, batch_size, shuffle=False, collate_fn=collate_fn)
    return dataloader


